from bs4 import BeautifulSoup

soup = BeautifulSoup("<html><body><p>data</p></body></html>")


class TableInfoExtractor:
    def __init__(self, html_str):
        self.html_str = html_str

    def get_dataframe_from_html_table(self):
        pass