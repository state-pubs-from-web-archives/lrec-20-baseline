from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import os 
import pandas as pd
import PyPDF2
import pdftotext
import pytesseract
import pandas as pd
import os
from pdf2image import convert_from_path
from PIL import Image
from absl import app
from tika import parser
from absl import flags
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "file_path", "...", "Path to the folder which contain folders")
flags.DEFINE_string(
    "tesseract_path", "...", "Path to tesseract.exe")
flags.DEFINE_string(
    "poppler_path", "...", "Path to poppler")
flags.DEFINE_string(
    "output_path", "...", "output path to store the dataset")



def ocr(file_path): #function to extract text using tesseract
    # try:
    text = ''

    ## add your own file_location to tesseract.exe
    # pytesseract.pytesseract.tesseract_cmd = FLAGS.tesseract_path


    ## add your own file_location to poppler
    images = convert_from_path(file_path+".pdf")
    for count, img in enumerate(images):
        extracted_text = pytesseract.image_to_string(img)
        text += extracted_text
        
        # adding this text for line break  
        text += '****************************************************************************************************'
        ## only saving first page information
        break
    text = str(text)
    text = " ".join(text.split())
    text = " ".join(text.split("\t"))
    return text
    # except:
    #     return 0
    


def extract_text_from_pdf(file_path): # function to extract text using pdfminer
    text = ''
    title = ''
    layout = ''
    # for page_layout in extract_pages(file_path):#+'.pdf'):
    for page_layout in extract_pages(file_path+'.pdf'):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    try:
                        text += text_line.get_text()
                    except:
                        continue
        text = str(text)
        text = " ".join(text.split())
        text = " ".join(text.split("\t"))
        if len(text) != 0:  
            
            # adding this text for line break                 
            text += '****************************************************************************************************'
            break
    with open(file_path+'.pdf', 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        try:
            title = reader.metadata['/Title']
        except:
            title = 'NA'
    title = str(title)
    title = " ".join(title.split())
    title = " ".join(title.split("\t"))
    
    return text, title, layout


def structural_features(data_file):
    title = list()
    total_chracter_l = list()
    total_words_l = list()
    total_lines_l = list()
    avg_words_page_l = list()
    avg_lines_page_l = list()
    space_char_per_l = list()
    avg_alphanum_per_l = list()
    line_ratio_l = list()
    total_upcase_page_l = list()
    total_alphnum_line_l = list()
    #extracting structural faetures based above extracted text
    for text in data_file['text']:
        page_count = 0
        total_lines = 0
        total_words = 0
        total_chracter = 0
        total_spaces = 0
        total_alphnum_page = 0
        total_upcase_page = 0
        total_alphnum_line = 0
        max = None
        min = None
        for page in text.split('****************************************************************************************************'):
            num_lines_page = 0
            num_words_page = 0
            num_char_page = 0
            num_spaces_page = 0
            num_alphnum_page = 0
            num_upcase_page = 0
            num_alphnum_line = 0
            if len(page) != 0 :
                for line in page.split('\n'):
                    if len(line) != 0 :
                        #print(len(line.split(" ")))
                        m = line.split(" ")
                        if(line[0].isupper()):
                            num_upcase_page += 1
                            
                        num_spaces_page += len(m)-1
                        num_char_page += len(line)
                        num_words_page += len(m)
                        num_lines_page += 1
                        
                        if max==None and min == None:
                            max = len(line)
                            min = len(line)
                        if len(line) > max:
                            max = len(line)
                        if len(line) < min:
                            min = len(line)
                            
                        for i in m:
                            if(i.isalpha() or i.isnumeric()):
                                continue
                            else:
                                if(i.isalnum()):
                                    num_alphnum_page += 1
                                    
                        if(line[0].isalpha() or line[0].isnumeric()):
                                continue
                        else:
                            if(line[0].isalnum()):
                                num_alphnum_line += 1
                page_count += 1
                                    
                total_alphnum_line += num_alphnum_line
                total_upcase_page += num_upcase_page
                total_alphnum_page += num_alphnum_page
                total_lines += num_lines_page
                total_words += num_words_page
                total_chracter += num_char_page
                total_spaces += num_spaces_page
        try:
            avg_lines_page = total_lines/page_count
            avg_words_page = total_words/page_count
            space_char_per = round((total_spaces / total_chracter) * 100, 2)
            avg_alphanum_per = (total_alphnum_page / total_words) * 100
            line_ratio  = min/max
        except:
            total_chracter_l.append(0)
            total_words_l.append(0)
            total_lines_l.append(0)
            avg_words_page_l.append(0)
            avg_lines_page_l.append(0)
            space_char_per_l.append(0)
            avg_alphanum_per_l.append(0)
            line_ratio_l.append(0)
            total_upcase_page_l.append(0)
            total_alphnum_line_l.append(0)
        total_chracter_l.append(total_chracter)
        total_words_l.append(total_words)
        total_lines_l.append(total_lines)
        avg_words_page_l.append(avg_words_page)
        avg_lines_page_l.append(avg_lines_page)
        space_char_per_l.append(space_char_per)
        avg_alphanum_per_l.append(avg_alphanum_per)
        line_ratio_l.append(line_ratio)
        total_upcase_page_l.append(total_upcase_page)
        total_alphnum_line_l.append(total_alphnum_line)
        
    data_file["total_chracter"] = total_chracter_l
    data_file["total_words"] = total_words_l
    data_file["total_lines"] = total_lines_l
    data_file["avg_words_page"] = avg_words_page_l
    data_file["avg_lines_page"] = avg_lines_page_l
    data_file["space_char_per"] = space_char_per_l
    data_file["avg_alphanum_per"] = avg_alphanum_per_l
    data_file["line_ratio"] = line_ratio_l
    data_file["total_upcase_page"] = total_upcase_page_l
    data_file["total_alphnum_line"] = total_alphnum_line_l
    
    data_file.to_csv(FLAGS.output_path+'data.csv') 


def main(argv):
    os.chdir(FLAGS.file_path)
    print(os.getcwd())
    p = os.listdir()
    c = list()
    num = 20
    count = 0
    df = pd.DataFrame(columns = ['pdf_name', 'text', 'title', 'layout'])
    file_to_write = open(FLAGS.output_path+"all_tsv.txt","w")
    idx = 0
    for k, i in enumerate(p[:]):
        for file in tqdm(os.listdir(os.path.join("./",i))):
            target_path = os.path.join("./",i,file,file)
            text, title, layout =  extract_text_from_pdf(target_path)#('./'+i+'/'+i)
            if text == 0 or text == '':
                text =  ocr(target_path)
                if text == 0 or text == '':
                    count += 1
                    continue
            df.loc[idx] = [i, text, title, layout]
            idx += 1
            text_to_write = str(idx) + "\t" + str(file) + "\t"+str(i) + "\t"+ str(text) + "\t" + str(title) + "\n"
            file_to_write.write(text_to_write)
    df.to_csv(FLAGS.output_path+'all.csv') 
    

if __name__ == "__main__":
    app.run(main)
