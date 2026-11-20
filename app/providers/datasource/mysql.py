import pymysql
import pymysql.cursors

from app.core.config import settings
from .base import DataSourceProvider


class MySQLDataSourceProvider(DataSourceProvider):
    def connect(self) -> pymysql.Connection:
        return pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",
            connect_timeout=settings.MYSQL_CONNECT_TIMEOUT,
        )
