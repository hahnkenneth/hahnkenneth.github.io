---
layout: page
title: Remote Work and Travelling
description: Data Analytics with Python
img: assets/img/wfh.jpg
report_pdf: Final Report.pdf
importance: 3
category: Python Programming
---

<header class="post-header">
    <h1 class="post-title">
        <a
        href="{{ page.report_pdf | prepend: 'assets/pdf/' | relative_url}}"
        target="_blank"
        rel="noopener noreferrer"
        class="float-middle"
        ><i class="fa-solid fa-file-pdf"></i
        ></a>
    </h1>
</header>


This was a data anlytics and visualization project on the impact of COVID, work from home (WFH) job postings, and both of their impacts on travelling (measured by the number of trips that people took). The data was taken from two sources, [the Trips by Distance dataset](https://catalog.data.gov/dataset/trips-by-distance) and [the WFH map](https://wfhmap.com/data/), where the Trips by Distance dataset measures the number of trips that were taken over time for each state and county while the WFH map shows data on WFH map shows the number of job postings by state. Our goal was to analyze various time trends on the state and national level to create a visual representation of how travelling has evolved due to COVID and the increase in job postings.

The results of the model as well as the assumptions and evaluations are in the pdf report above and below. The code, report, and presentation files can be found in the Github Repository [here](https://github.com/hahnkenneth/DataSci_200_WFH_Trips).

<iframe src="/assets/pdf/Final Report.pdf" width="100%" height="600px">
</iframe>