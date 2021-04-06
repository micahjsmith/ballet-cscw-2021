import re
import sys
from functools import update_wrapper
from pathlib import Path
from typing import Any, Iterator, List
from urllib3.util.retry import Retry

import click
import funcy as fy
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from github import Github
from sqlalchemy import desc, create_engine, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from stacklog import stacktime
from tqdm import tqdm

from models import Base, Repository, RepositoryLanguage, RepositoryTopic


Session = sessionmaker()


BAD_TOPICS = [
    'awesome',
    'interview',
]

BAD_REPOS = [
    'freeCodeCamp/freeCodeCamp',
]

EXTRA_DS_REPOS = [
    'Chicago/food-inspections-evaluation',
    'microsoft/CameraTraps',
    'Microsoft/speciesclassification',
    'microsoft/landcover',
    'open-mmlab/mmdetection',
    'facebookresearch/detectron2',
    'CoronaWhy/task-ts',
    'OHDSI/PatientLevelPrediction',
    'Data4Democracy/drug-spending',
    'dssg/police-eis',
    'dssg/peeps-chili',
    'dssg/infonavit-public',
    'dssg/voter-protection',
    'dssg/usal_echo_public',
    'dssg/barefoot-winnie-public',
    'dssg/IEFP-RecSys_public',
    'dssg/hiv-retention-public',
    'dssg/el-salvador-mined-public',
    'dssg/chile-dt-public',
    'tesseract-ocr/tesseract',
    'ageitgey/face_recognition',
    'deepfakes/faceswap',
    'CMU-Perceptual-Computing-Lab/openpose',
    'mozilla/STT',
    'JaidedAI/EasyOCR',
    'emilwallner/Screenshot-to-code',
    'esimov/caire',
    'esimov/pigo',
    'Hironsan/BossSensor',
    'liuruoze/EasyPR',
    'wb14123/seq2seq-couplet',
    'infinitered/nsfwjs',
]


def here() -> Path:
    return Path(__file__).parent.resolve()


def uniquify(seq, idfun=repr):
    """Source: https://code.activestate.com/recipes/52560/#c15"""
    seen = {}
    return [seen.setdefault(idfun(e), e) for e in seq if idfun(e) not in seen]


def get_token() -> str:
    return (
        Path
        .home()
        .joinpath('.github', 'token.txt')
        .open('r')
        .read()
        .strip()
    )


def github() -> Github:
    return Github(
        get_token(),
        retry=Retry(
            total=20,
            status_forcelist=(403, 500, 502, 504),
            backoff_factor=0.4,
        )
    )


def count_contributions(g, repo: Repository) -> int:
    if repo.id == 'torvalds/linux':
        return 10996  # github ui as of 2020-10-01T22:07:00Z
    result = count_contributions_scrape(g, repo)
    if result is None:
        result = count_contributions_api(g, repo)
    return result


@fy.silent
def count_contributions_api(g, repo: Repository) -> int:
    """Count User-type contributors using v3 API"""
    return (
        g
        .get_repo(repo.id)
        .get_contributors()
        .totalCount
    )


@fy.silent
@fy.post_processing(int)
@fy.post_processing(lambda s: s.replace(',', ''))
def count_contributions_scrape(g, repo: Repository) -> int:
    """Count contributors as they approximately appear on repo page

    On the repo page, the number of contributors is displayed in a badge. If the number is greater than some threshold (which appears to be 5000), then "5,000+" is displayed. You can get the count in this case by counting the elements of the unordered list of profiles that is shown plus the number of contributors linked to in the supplemental link. We don't implement all of that functionality here, in this case we don't count the list of profiles so it is just an approximate count.
    """
    url = f'https://github.com/{repo.id}'
    response = requests.get(url)
    response.raise_for_status()
    body = response.text
    soup = BeautifulSoup(body, 'html.parser')

    tags = soup.find_all('a', href=re.compile('graphs/contributors'))
    for tag in tags:
        if child := tag.find('span', class_='Counter'):
            # not accurate count if shows "5,000+"
            if '+' not in child.text:
                return child.text

        if text := fy.re_find(r'.* (\d[,\d]+) contributors.*', tag.text):
            return text

    return None



def get_interesting_repos(g: Github, session: Any) -> List[Repository]:
    repos: List[Repository] = []

    grepos = g.search_repositories(
        query='stars:>250 forks:>50', sort='stars', order='desc')
    records = zip(grepos, fy.repeat('most_stars'))

    grepos = g.search_repositories(
        query='forks:>5 topic:kaggle-competition', sort='stars', order='desc')
    records = fy.concat(records, zip(grepos, fy.repeat('kaggle')))

    grepos = g.search_repositories(
        query='forks:>5 topic:tensorflow-model', sort='stars', order='desc')
    records = fy.concat(records, zip(grepos, fy.repeat('tensorflow-model')))

    grepos = g.search_repositories(
        query='cookiecutterdatascience in:readme forks:>5 stars:>0 fork:true',
        sort='stars', order='desc')
    records = fy.concat(
        records, zip(grepos, fy.repeat('cookiecutterdatascience')))

    for grepo, search_method in tqdm(records):
        repo = (
            session
            .query(Repository)
            .filter(Repository.id == grepo.full_name)
            .one_or_none()
        )
        if repo is None:
            repo = Repository(
                id=grepo.full_name,
                owner=grepo.owner.login,
                name=grepo.name,
                description=grepo.description,
                search_method=search_method,
            )
            repos.append(repo)

    return repos


def add_specific_repos(
    g: Github, session: Any, fullnames: List[str]
) -> Iterator[Repository]:
    for fullname in fullnames:
        repo = (
            session
            .query(Repository)
            .filter(Repository.id == fullname)
            .one_or_none()
        )
        if repo is None:
            grepo = g.get_repo(fullname)
            repo = Repository(
                id=grepo.full_name,
                owner=grepo.owner.login,
                name=grepo.name,
                description=grepo.description,
                search_method='custom',
            )
            yield repo


def populate_stars_languages_topics(g: Github, repos: List[Repository]):
    for repo in repos:
        grepo = g.get_repo(repo.id)
        repo.stars = grepo.stargazers_count
        repo.forks = grepo.forks
        repo.languages = [
            RepositoryLanguage(
                repository_id=grepo.full_name,
                language=language,
            )
            for language in grepo.get_languages()
        ]
        repo.topics = [
            RepositoryTopic(
                repository_id=grepo.full_name,
                topic=topic,
            )
            for topic in grepo.get_topics()
        ]


def is_eligible_repo(g, repo: Repository) -> bool:
    return all([
        len(repo.languages) > 0,
        all(
            term not in item.topic
            for item in repo.topics
            for term in BAD_TOPICS
        ),
        repo.id not in BAD_REPOS,
    ])


@click.group()
@click.option('--pdb', is_flag=True)
@click.pass_context
def cli(ctx, pdb):
    ctx.ensure_object(dict)
    ctx.obj['pdb'] = pdb


def postmortem(f):
    @click.pass_context
    def decorated(ctx, *args, **kwargs):
        try:
            return ctx.invoke(f, *args, **kwargs)
        except Exception as e:
            if not isinstance(e, click.ClickException):
                if ctx.obj.get('pdb', False):
                    import pdb
                    pdb.post_mortem()
            raise
    return update_wrapper(decorated, f)


def initdb():
    engine = create_engine('sqlite:///data/main.db')
    Base.metadata.create_all(engine)
    Session.configure(bind=engine)


def load_software_table():
    df_software = pd.read_csv('data/largest_collaborations_most_stars.csv', index_col=0)
    df_software = df_software[df_software['eligible']]
    df_software.loc[df_software['id'] == 'torvalds/linux', 'contributors'] = 20000
    df_software = df_software.dropna(subset=['contributors'])
    df_software['contributors'] = df_software['contributors'].astype(int)
    df_software = df_software.sort_values('contributors', ascending=False)
    return df_software


def load_ds_table():
    redlist = [
        'open-mmlab/mmdetection',
        'facebookresearch/detectron2',
    ]
    df_datascience = pd.read_csv('data/largest_collaborations_datascience.csv', index_col=0)
    df_datascience = df_datascience[~df_datascience['id'].isin(redlist)]
    return df_datascience


def make_comparison_table(df_software, df_datascience, n=10):
    df = df_software[['id', 'contributors']].assign(kind='software')
    df = df.append(
        df_datascience[['id', 'contributors']].assign(kind='datascience')
    )
    table = (
        df
        .sort_values('contributors', ascending=False)
        .groupby('kind')
        .apply(lambda group: group.iloc(axis=0)[:n])
        .reset_index(drop=True)
    )
    table_ds = table.query('kind == "datascience"').drop('kind', axis=1).reset_index(drop=True)
    table_se = table.query('kind == "software"').drop('kind', axis=1).reset_index(drop=True)
    table_se['contributors'] = (
        (
            (table_se['contributors']/100)
            .apply(np.floor)*100
        )
        .astype(int)
        .apply(lambda x: f'{x:,}+')
    )
    table = table_se.join(table_ds, rsuffix='x')
    return table


@cli.command()
@click.option('--get-interesting/--no-get-interesting',
              is_flag=True, default=True)
@click.option('--add-specific-repos/--no-add-specific-repos', 'add_specific',
              is_flag=True, default=False)
@click.option('--update-stars-etc/--no-update-stars-etc',
              is_flag=True, default=True)
@click.option('--filter-eligible/--no-filter-eligible',
              is_flag=True, default=True)
@click.option('--count-contributors/--no-count-contributors',
              is_flag=True, default=True)
@postmortem
def load(
    get_interesting: bool,
    add_specific: bool,
    update_stars_etc: bool,
    filter_eligible: bool,
    count_contributors: bool,
) -> List[Repository]:
    initdb()
    session = Session()
    g = github()

    if get_interesting:
        with stacktime(print, 'Getting interesting repos'):
            repos = get_interesting_repos(g, session)
            repos = uniquify(repos, idfun=lambda repo: repo.id)
            session.add_all(repos)
            session.commit()

    if add_specific:
        with stacktime(print, 'Adding other specific repos'):
            repos = list(add_specific_repos(g, session, EXTRA_DS_REPOS))
            for repo in repos:
                try:
                    session.add(repo)
                    session.commit()
                except IntegrityError:
                    session.rollback()

    if update_stars_etc:
        with stacktime(print, 'Updating stars, languages, and topics'):
            repos = session.query(Repository).all()
            populate_stars_languages_topics(g, repos)
            session.add_all(repos)
            session.commit()

    if filter_eligible:
        with stacktime(print, 'Filtering eligible repos'):
            repos = session.query(Repository).all()
            for repo in tqdm(repos):
                repo.eligible = is_eligible_repo(g, repo)
            session.add_all(repos)
            session.commit()

    if count_contributors:
        with stacktime(print, 'Counting contributors'):
            repos = session.query(Repository).all()
            for repo in tqdm(repos):
                if repo.eligible:
                    repo.contributors = count_contributions(g, repo)
            session.add_all(repos)
            session.commit()

    return repos


@cli.command()
def analyze():
    initdb()
    session = Session()

    with stacktime(print, 'Analyzing most starred repos'):
        query = (
            session
            .query(Repository)
            .filter(Repository.eligible)
            .filter(Repository.search_method == 'most_stars')
            .order_by(desc(Repository.contributors))
        )
        result = query.all()
        df = (
            pd.DataFrame.from_records(fy.pluck_attr('__dict__', result))
            .drop('_sa_instance_state', axis=1)
        )
        df.to_csv('./data/largest_collaborations_most_stars.csv')

    with stacktime(print, 'Analyzing data science projects'):
        query = (
            session
            .query(Repository)
            .filter(Repository.eligible)
            .filter(
                or_(Repository.search_method != 'most_stars',
                    Repository.id.in_(EXTRA_DS_REPOS))
            )
            .order_by(desc(Repository.contributors))
        )
        result = query.all()
        df = (
            pd.DataFrame.from_records(fy.pluck_attr('__dict__', result))
            .drop('_sa_instance_state', axis=1)
        )
        df.to_csv('./data/largest_collaborations_datascience.csv')

    with stacktime(print, 'Creating comparison table'):
        df_software = load_software_table()
        df_datascience = load_ds_table()
        df_comparison = make_comparison_table(df_software, df_datascience)
        result = df_comparison.to_latex(
            buf=None, header=False, index=False, float_format='%1.0f')
        with open('project_sizes_tabular.tex', 'w') as f:
            f.write('\n'.join(result.split('\n')[2:][:-3]))
            f.write('\n')


if __name__ == '__main__':
    sys.exit(cli())
