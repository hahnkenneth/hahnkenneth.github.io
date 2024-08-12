---
layout: page
title: Blood Pressure Risk Factors
description: Linear Regression with R
img: assets/img/blood.jpg
importance: 2
category: Statistics
report_pdf: Final-Report.pdf
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

This project was an analysis on Blood Pressure based on data from the [National Health and Nutrition Examination Survey](https://wwwn.cdc.gov/Nchs/Nhanes/Search/DataPage.aspx?Component=Questionnaire&%20Cycle=2013-2014), a survey that combines both interview data along with physical examinations to characterize the prevalence of major diseases. We evaluate the significance of both the respondent's body weight as well as their proximity to smoking in order to create a descriptive linear regression model for one's Mean Arterial Blood Pressure.

The results of the model as well as the assumptions and evaluations are in the pdf report above. The data and the R Markdown file to create the report can be found in the Github Repository [here](https://github.com/hahnkenneth/DataSci_203_Blood_Pressure_Linear_Regression/tree/main).

<iframe src="/assets/pdf/Final-Report.pdf" width="100%" height="600px">
</iframe>
