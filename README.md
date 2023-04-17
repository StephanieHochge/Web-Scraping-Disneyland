# A Web Scraping Approach to Inform Theme Park Planning

This project contains all scripts created in the course DLBDSDQDW01 to work on Task 2 "Scrape the Web". The aim of the project was to use web scraping to create a data base to inform key business decisions for a company planning a new theme park. The company does not yet have expertise in this area, so it starts looking at the wait times of popular attractions at Disneyland Paris with the aim of determining factors that influence the wait times to enable informed decisions to be made. For this purpose, it also looks at weather forecasts for different time periods in Paris.

Below is a brief description of the files used to work on this project.
- hdf5_data_model.py: This file contains the code to create the HDF5 data model used to store the scraped wait times and weather forecasts.
- scrape_rides.py: This file contains the code to extract wait times from two websites that display wait times in near real time from attractions at various parks, including Disneyland.
- scrape_weather.py: This file contains the code to extract weather forecasts for Paris for three different time periods (the current day, the next day and next week).
- csv_to_hdf5.py: This file contains the code to store the wait times in the HDF5 file.
- global_functions.py: This file contains functions used multiple of the other scripts.
- disneyland.hdf5: This HDF5 file contains data (wait times and weather forecasts) extracted from April 2, 2023 to April 14, 2023 using the aforementioned scripts.
- scrape_disneyland.log: This log file recorded important events during data collection, including the storage of data in the HDF5 file and any errors encountered during web scraping. 
- test_scraping.py: This file contains a script to test the main functions of the aforementioned scripts.
- visualization.ipynb: This notebook contains the code to create visualizations of the collected data.
- requirements.txt: This file contains the project dependencies. 
- websites: This file contains the links to the scraped websites. 
