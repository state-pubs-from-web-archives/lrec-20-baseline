# Readme for scripts and dataset for last grant

## Folder Overview :

### These folders include 1 folder conatining sample datasets and another one is python notebook

1. text_structural_features.ipynb --- This notebook contains the script to extract text as well as 11 structural features from the pdf.
2. text_structural_features.py --- This script has the same functionality as script in the notebook(text_structural_features.ipynb).
3. sample_data_set -- This folder 2 sample datasets which are generated from UNT.edu repository
	- st_p.csv --- This is positive dataset generated from UNT.edu repository
	- st_n.csv --- This is negative dataset generated from UNT.edu repository


### Script Running :

1. text_structural_features.ipynb :
	Just use any notebook supported IDE like jupyter notebook, google colab[ for colab you need to mount your drive for pdf files], VScode etc.
	while running the notebook use own file path for pdf files, file location for tesseract.exe as well as for poppler.

	If unable to download the poppler and tesseract.exe use pytesseract module and python-poppler module.

2. text_structural_features.py : 
	In order to compile the script you need to provide 4 different options which are
	a. file_path -- path to the folder which contain pdf files
	b. tesseract_path -- path to tesseract.exe
	c. poppler_path -- path to poppler folder
	d. output_path -- path to store the output dataframe(.csv format).

	ex : python3 text_structural_features.py --file_path data/full_pdf_data/1.Texas_Pub_Candidates/ --tesseract_path ../tesseract/tesseract.exe --poppler_path ../poppler/ --output_path ../..



If you find errors or have questions please contact
Mark Phillips at mark.phillips@unt.edu or Praneeth Rikka at praneethrikka@my.unt.edu


