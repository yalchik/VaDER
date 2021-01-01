import sys
import logging
from logging import Formatter
from typing import Dict, Optional


class LogManager:
    """Configures and holds loggers and provides the interface to get a certain logger by name"""

    FORMATTER: Formatter = Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

    def __init__(self):
        self.loggers_dict: Dict[str, logging.Logger] = {}

    def get_logger(self, logger_name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Returns a certain logger by its name if it exists, otherwise configures a new one.
        If a log_file is specified, logging will be duplicated in stdout and the file.
        If a log_file is not specified, the logs will be written in stdout only.

        Parameters
        ----------
        logger_name : str
            Usually equals to __name__ in a calling function.
        log_file : str or None
            File path, where the logs will be written. If None, no logs will be written in files.

        Returns
        -------
        A logger object
        """
        if logger_name in self.loggers_dict:
            logger = self.loggers_dict[logger_name]
        else:
            logger = self.__create_logger(logger_name, log_file)
            self.loggers_dict[logger_name] = logger
        return logger

    def __create_logger(self, logger_name: str, log_file: Optional[str] = None):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.__get_console_handler())
        if log_file:
            logger.addHandler(self.__get_file_handler(log_file))
        logger.propagate = False
        return logger

    def __get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.FORMATTER)
        return console_handler

    def __get_file_handler(self, log_file: str):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self.FORMATTER)
        return file_handler
