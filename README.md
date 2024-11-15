# search-engine
The third project for my Information Retrieval and Web Search course. I built a small scale search engine to process conjunctive and disjunctive queries against an index built on a Reuters corpus.

## CODE

Any code for the project can be found in the /src/ subdirectory. It contains code for all subprojects. It can be ran directly from the .py files. These details are elaborated
on in /Reports/40133600_demo.pdf.

The /Corpus/ subdirectory contains the corpus for this project. If you wish to refresh its contents, please remove everything and fill it with the 
contents of the reuters21578.tar.gz file EXCLUSIVELY.

## REPORTS

Both the report and demo are found in the /Reports/ subdirectory. 

## SUBPROJECT 1

This subproject compares the performance differences in the index construction phase, between the Naïve and SPIMI index construction methods. Further details can be found in the report under /reports/40133600_report.pdf.

#### SUBPROJECT 1 RESULTS

The results to subproject 1 can be found in the 40133600_report.pdf file, under the "Statistical Performance" subsection which can be accessed directly via the link in the table of contents.

## SUBPROJECT 2

- This subproject is devoted to building a query processing layer on top of the index, allowing for simple conjunctive and disjunctive queries to be ran against the index.
- The results returned are ranked, according to the ranking method selected. Further details on the ranking methods can be found in the report under /reports/40133600_report.pdf.

#### SUBPROJECT 2 QUERY RUNS

Output for the queries can also be found in 40133600_report.pdf, in both Appendix A and Appendix B. They are elaborated on in the subsections preceding the appendices.

## OUTPUT

NOTE: All output generated for this project is done so via the terminal output. 