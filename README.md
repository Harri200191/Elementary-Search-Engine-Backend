# Elementary Search Engine

## Data Set Details

For the project, we’ll be using the nela-gt-2022 dataset which contains myriads of articles on various topics across STEM (Science, Technology, Engineering, Mathematics). It will serve as a good dataset for the semester project and also fulfill the requirements. 

## Frame Works and Tools

We have chosen python for this project for its fast prototyping ability and rich ecosystem of tools and libraries. Our lexicon is being populated as our articles are being forward indexed. After the processing, all the processed indexes are stored in .JSON files. Before being added to the lexicon, all the words are lemmatized for better search experience. We preferred lemmatization technique over stemming for its better result accuracy and preservation of the meaning of the words. For our inverted index, a list is created against each word_id (loaded from forward index) and doc_ids are appended to it.

Our frontend is made in ReactJS as it is the leading industry standard.
Moreover, we can also deploy the frontend and the servers on cloud platforms and make the search engine available to all, however for the project, we plan to do it all on localhost.
