---
layout: page
title: Voting Difficulty and Political Party
description: Hypothesis Testing with R
img: assets/img/voting.jpg
report_pdf: Final_Report.pdf
importance: 3
category: Statistics
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

This project was an analysis on Voting Difficulty based on data from the [American National Election Studies (ANES) 2022 Pilot Study](https://www.pewresearch.org/politics/2023/07/12/voter-turnout-2018-2022/), an observational dataset based on sample respondents collected from YouGov. With this dataset, we evaluate whether Democratic or Republican voters experience more difficulty voting using a Wilcoxon Rank-Sum Test.

The results of the hypothesis test as well as the assumptions and evaluations are in the pdf report above. The data and the R Markdown file to create the report can be found in the Github Repository [here](https://github.com/hahnkenneth/DataSci_203_VotingDifficulty_vs_PoliticalParty).

<iframe src="/assets/pdf/Final_Report.pdf" width="100%" height="600px">
</iframe>
