---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Marketing Engagement

# What are the common metrics to test and track engagement of the user?

Common metrics to test and track user engagement include:

- **Daily Active Users (DAU), Weekly Active Users (WAU), Monthly Active Users (MAU):** Measures how many unique users actively engage with the product in specific time windows.
- **DAU/MAU Ratio ("Stickiness"):** Indicates how often monthly users return daily; a higher ratio means stronger engagement and retention.
- **Average Session Length:** The amount of time a user spends per session interacting with the product.
- **Average Number of Sessions per User:** How frequently a user returns and engages in a given period.
- **Feature Activation Rate:** Percentage of users using a specific feature, reflecting feature adoption.
- **Conversion Rate:** Percentage of users completing a desired action like sign-ups, purchases, or downloads.
- **Customer Satisfaction Score (CSAT) and Net Promoter Score (NPS):** Surveys assessing user satisfaction and loyalty.
- **Churn Rate:** Rate at which users stop using the product.
- **Pages per Session or Screens per Visit:** Depth of user interaction in each visit.
- **Funnel Completion Rate:** Percentage of users completing a defined multi-step process.

These metrics help understand user behavior, retention, satisfaction, and product utility, forming the basis for optimizing the user experience and business growth.

# User lifecycle prediction

User lifecycle prediction refers to the process of forecasting the various stages a user will go through while interacting with a product or service. It aims to understand and anticipate user behavior from the moment they first become aware of a product until they either become loyal customers or disengage.

The user lifecycle typically includes stages such as:

- Acquisition: attracting new users

- Activation: ensuring users start using the product actively

- Retention: keeping users engaged over time

- Referral: encouraging users to recommend to others

- Revenue: converting users into paying customers and maximizing lifetime value

User lifecycle prediction leverages data and analytics to forecast how users will progress through these stages, enabling businesses to target marketing, engagement, and retention strategies more effectively. This helps optimize user experience, increase customer lifetime value, and foster long-term loyalty by proactively addressing user needs and behaviors at each phase

## Create a notebook with MyST Markdown

MyST Markdown notebooks are defined by two things:

1. YAML metadata that is needed to understand if / how it should convert text files to notebooks (including information about the kernel needed).
   See the YAML at the top of this page for example.
2. The presence of `{code-cell}` directives, which will be executed with your book.

That's all that is needed to get started!

## Quickly add YAML metadata for MyST Notebooks

If you have a markdown file and you'd like to quickly add YAML metadata to it, so that Jupyter Book will treat it as a MyST Markdown Notebook, run the following command:

```
jupyter-book myst init path/to/markdownfile.md
```
