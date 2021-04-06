from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta


Base: DeclarativeMeta = declarative_base()


class Repository(Base):
    __tablename__ = 'repositories'

    id = Column(String, primary_key=True)
    owner = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    search_method = Column(String, nullable=False)
    stars = Column(Integer, nullable=True)
    forks = Column(Integer, nullable=True)
    eligible = Column(Boolean, default=True, nullable=False)
    contributors = Column(Integer, nullable=True)
    languages = relationship(
        'RepositoryLanguage',
        order_by='RepositoryLanguage.id',
        back_populates='repository')
    topics = relationship(
        'RepositoryTopic',
        order_by='RepositoryTopic.id',
        back_populates='repository')


class RepositoryLanguage(Base):
    __tablename__ = 'repositorylanguages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey('repositories.id'))
    repository = relationship(
        'Repository', back_populates='languages')
    language = Column(String, nullable=False)


class RepositoryTopic(Base):
    __tablename__ = 'repositorytopics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey('repositories.id'))
    repository = relationship(
        'Repository', back_populates='topics')
    topic = Column(String, nullable=False)
