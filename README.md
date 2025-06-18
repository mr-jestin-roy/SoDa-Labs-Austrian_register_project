# Optical character recognition on historical handwritten texts

Repository for the Austrian Register Data Project 📖

---
 
**Given**: Dataset of historical handwritten text from Austrian birth, baptism, marriage, and death registers. Archives are present in tabular form (table being hand-drawn or block-printed) and text is cursive handwriting. [Metricula online dataset](https://data.matricula-online.eu/en/bestande/).

**Overview**

- Text dates 1625-2006. More info [here](https://data.matricula-online.eu/en/allgemeine-infos/).  
- 183,312 church registers; 4,502,868 images scraped and stored on AWS S3.

Sample ground truth:  
![sample ground truth](./assets/images/04-Trauung_0005.jpg)

> **Notes on dataset:**  
- Tables are not uniform in structure (variation in fields/columns), owing to the range of periods and geographic locations the dataset comes from
- Handwriting styles vary due to differences in human-writers and periods
- Noise is present in the form of ink blots, parchment texture, or general wear and tear of the physical documents

> Matricula images were scraped using [austrian_register_project/Scraper at main · sodalabsio/austrian_register_project (github.com)](https://github.com/sodalabsio/austrian_register_project/tree/main/Scraper); originally <https://github.com/1fge/matricula-online-scraper>

> Sample Matricula dataset present at: https://drive.google.com/drive/folders/15eqsj8LIIz9bv_VLS8pzlua7WvbymnSn?usp=sharing

***Full dataset**: on AWS S3 bucket “austrian-register-data”. Ask access from Satya Borgohain. (Some part of it) transferred to Paul’s M3; 496 images at /projects/oc23/mini_images/ and transcriptions at /projects/oc23/austrian_images.txt.*

---
