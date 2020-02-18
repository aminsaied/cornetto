#!/usr/bin/env python3
"""Script collecting data from arxiv.org.

Collects metadata of all the papers published between two given days (both
days given as YYYY-MM-DD strings). The metadata we collect is:

  -- 'arXiv_id' , unique id, of the format 'int.int' or just 'int'
  -- 'Title' , the title of the paper
  -- 'Authors' , as a string of last and first names separated by a delimiter
  -- 'Date' , date of publication, YYYY-MM-DD
  -- 'MSCs' , string of 5-character codes, separated by a delimiter.
  -- 'Abstract' , short description of the content of the paper

The data is saved as pandas DataFrame to '../data/arxiv/'.
"""
import datetime
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.arxiv_harvester import ArxivHarvester

def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date should be in the form YYYY-MM-DD")

if __name__ == '__main__':
    print("Harvesting arxiv.org. Note: this process takes a while, ")
    print("because of the necessary pause between server requests. ")
    print("--------------------------------------------------------")
    print("Enter the start date as YYYY-MM-DD. Default is 2010-01-01.")
    print("(Press enter to use the default)")
    start_date = input()
    print("Enter the end date as YYYY-MM-DD. Default is today.")
    print("Just press enter to use the default.")
    end_date = input()

    print("Starting the harvest...")
    harvester = ArxivHarvester()
    harvester.harvest(start_date=start_date, end_date=end_date)
    print("The harvest was successful!")
