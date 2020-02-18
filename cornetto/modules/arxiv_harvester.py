#!/usr/bin/env python3
"""Collects metadata from the arxiv.

Contains :class:`ArxivHarvester` for scraping metadata from arxiv papers.
The arxiv is part of the "Open Archives Initiative" which facilitates the
harvesting of metadata.
"""
import re, time
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

START_URL_TEMPLATE = ("http://export.arxiv.org/oai2?verb=ListRecords&set=math"
                      "&metadataPrefix=arXiv&from=%s&until=%s"
                     )
NEXT_URL_TEMPLATE  = ("http://export.arxiv.org/oai2?verb=ListRecords&resumpti"
                      "onToken=%s"
                     )
ARXIV_DATA_PATH = "../data/arxiv_raw/"
NAME_TEMPLATE = 'arxiv_%s.pkl'
DELIMITER = "|"

class ArxivHarvester(object):
    """Class designed to collect metadata on math papers from arxiv.org."""
    def __init__(self, arxiv_data_path=ARXIV_DATA_PATH,
                       name_template=NAME_TEMPLATE):
        self.arxiv_data_path = arxiv_data_path
        self.name_template = name_template

    def harvest(self, start_date=None, end_date=None):
        """Goes to the arxiv and asks for its data.

        Collects the data from arxiv.org into pandas dataframes and saves the
        dataframes into the desired folder. By default the files are saved
        into "../data/arxiv/". Each dataframe contains 1000 records (this is
        how much arxiv.org returns at a time.)

        Args:
            start_date, str: format 'YYYY-MM-DD'.
            end_date, str: format 'YYYY-MM-DD', defaults to today's date.

        Note:
            This method takes a while to run due to necessary time
            gaps between succesive server requests.
        """
        if not start_date:
            start_date='2010-01-01'
        if not end_date:
            end_date=time.strftime("%Y-%m-%d")

        url = START_URL_TEMPLATE%(start_date, end_date)
        df = pd.DataFrame(columns=('arXiv_id','Title','Authors', 'Date', 'MSCs','Abstract'))

        file_counter = 1

        while True:
            r = self.__class__._make_request(url)
            if not r:
                print("HTTP Error, stoping the harvesting.")
                break
            print("Processing request number %s..."%file_counter)

            soup = bs(r.text,'lxml')
            for entry in soup.find_all('record'):
                try:
                    metadata = self.__class__._get_metadata(entry)
                    df.loc[df.shape[0]] = metadata
                except AttributeError as e:
                    continue

            # pickle the data-frame with these 1000 papers
            df.to_pickle(self.arxiv_data_path+self.name_template%file_counter)
            file_counter += 1

            res_token =  soup.find("resumptiontoken")
            if not res_token:
                print("Done harvesting.")
                break
            url = NEXT_URL_TEMPLATE%res_token.string
            time.sleep(60)

    @staticmethod
    def _make_request(url):
        """Requests the given url and returns the response.

        In case of HTTP Error, sleeps 600 seconds (this is how much arxiv.org
        wants). If the error repeats more than 3 times, it stops trying and
        returns.

        Args:
            url, str: The URL where the request is to be made.
        """
        counter = 0
        MAX_COUNTER = 3
        while counter < MAX_COUNTER:
            try:
                r = requests.get(url)
                return r
            except requests.exceptions.HTTPError as e:
                print("HTTP Error. Waiting 600 sec before ")
                counter += 1
                time.sleep(601)
                continue
            except requests.exceptions.RequestException as e:
                raise

    @staticmethod
    def _get_metadata(soup_entry):

        xstr = lambda s: '' if s is None else str(s)

        arXiv_id = xstr(soup_entry.id.string)
        title = xstr(soup_entry.title.string)

        full_name = lambda author: xstr(author.keyname.string) + ',' + xstr(author.forenames.string)
        full_names = [full_name(author) for author in soup_entry.authors]
        authors = DELIMITER.join(full_names)

        date = xstr(soup_entry.datestamp.string)

        msc_classes = soup_entry.find('msc-class')
        if msc_classes is not None:
            mscs = re.findall(r'\d\d\w\d\d', msc_classes.string )
        else:
            mscs = []
        subjects = DELIMITER.join([msc for msc in mscs])

        abstract = xstr(soup_entry.abstract.string)

        metadata = {
            'arXiv_id': arXiv_id ,
            'Title': title ,
            'Authors': authors ,
            'Date': date ,
            'MSCs': subjects ,
            'Abstract': abstract
        }

        return metadata
