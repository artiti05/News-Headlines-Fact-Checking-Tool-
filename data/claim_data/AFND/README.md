# AFND

Arabic Fake News Dataset (AFND) is a collection of public Arabic news articles that were collected from public Arabic news websites. It contains 606912 news articles collected from 134 different public Arabic news websites. Misbar, which is a public Arabic news fact check platform, is used to classify the articles into credible, not credible, and undecided.


The file named "sources.json" contains 134 lines which corresponds to 134 public Arabic news websites. The Uniform Resource Locator (url) of the public Arabic news websites are replaced with "source_1", "source_2", "source_3", etc. to guarantee the anonymity of these websites.


The scraped public Arabic articles are stored in 134 sub-directories within the directory named "Dataset". Each sub-directory is named according to the anonymous names (e.g. "source_1") of the 134 news sources. Each sub-directory has a file named "scraped_articles.json" which contains the information of the articles that were scraped from the public news source and stored as an array of JSON objects. Each object stores the title, text, and the publication date of the article.



Contributors: Ashwaq Khalil, Moath Jarrah, and Monther Aldwairi

