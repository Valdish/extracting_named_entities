import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import re
import sqlite3
import docx2txt
import time
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    DatesExtractor,
    AddrExtractor,
    Doc
)
from io import BytesIO
import PyPDF2
import base64


# создание боковой панели
page = st.sidebar.selectbox("", ["Работа с текстовым сообщением", "Работа с текстовыми файлами", "Работа с файлами PDF",
                                 "База данных"])
page_bg_img = '''
<style>
 h1,h2 {
    
    padding: 10px;
    background-color:#545454;
    text-align:center; color: white;
   }
.stApp {

  background-size: cover;
}
h3,h4,h5,h6,p {text-align:center; color: white;}

</style><h1>Система поиска и обработки именованных сущностей</h1>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

start_time = time.time()

# создание экстракторов для извлечения сущностей
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
address_extractor = AddrExtractor(morph_vocab)

def Button():
    my_button = st.markdown("""
            <style >
            div.stButton > button: {
                position: absolute;
            }
            div.stButton > button:first-child {
                background-color: #578a00;
                color:#ffffff;
            }
            div.stButton > button:hover {
                background-color: #00128a;
                color:#ffffff;
                }
            </style>""", unsafe_allow_html=True)
    my_button = st.button('Обезличить файл')
    return  my_button

# таблица именованных сущностей
def Create_table_entities(text, type):
    conn = sqlite3.connect("soob.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS named_entities
                      (entity_text TEXT COLLATE NOCASE, entity_type TEXT COLLATE NOCASE)''')
    cursor.execute("INSERT INTO named_entities (entity_text, entity_type) VALUES (?, ?)", (text, type))
    conn.commit()
    conn.close()

newfilename = 'result.docx'
def writefile(t):
    f = open(newfilename, 'w')
    f.write(t)
    f.close()
    print("File written.")

# извлекатель дат
def extract_and_store_entities(soob):
    matches = dates_extractor(soob)
    for match in matches:
        start, stop = match.start, match.stop
        list_entities.append(soob[start:stop])
        Create_table_entities(soob[start:stop], "dates")

        # Extract addresses and add to the database
        # if span.type == 'AddressType':  # Replace 'AddressType' with the actual type used for addresses
        #     Create_table_entities(span.text, 'Address')

# извлекатель адресов
def extract_address(soob):
    matches = address_extractor(soob)
    for match in matches:
        address = soob[match.start:match.stop]
        list_entities.append(address)
        Create_table_entities(address, "address")


def calculate_f1(person, anonymized):
    # Function to calculate precision, recall, and F1 score
    original_entities = set(person) # Assuming you're looking for word entities
    st.text(person)
    anonymized_entities = set(anonymized)

    true_positive = len(original_entities)
    false_positive = len(anonymized_entities - original_entities)
    false_negative = len(original_entities - anonymized_entities)
    st.text(true_positive) # 1
    st.text(false_positive) # 10
    st.text(false_negative) # 8

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0 # 0.09
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0  # 0.11
    st.text(f"precision: {precision:.3f}")  # 10
    st.text(f"recall: {recall:.3f}")

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1_score

def calculate_accuracy(person, anonymized):
    original_entities = set(person)
    anonymized_entities = set(anonymized)

    correct_entities = len(original_entities & anonymized_entities)
    total_entities = len(original_entities)

    accuracy = correct_entities / total_entities if total_entities > 0 else 0.0

    return accuracy

###################################################################################################################
if page == "Работа с текстовыми файлами":
    uploaded_file = st.file_uploader('Загрузка файла', type='docx')

    # ФАЙЛ
    if Button():

        filename = uploaded_file.name

        con = sqlite3.connect("soob.db")
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS start (txt TEXT)")
        cur.execute('SELECT COUNT(*) from start')
        cur_result = cur.fetchone()

        rows = cur_result[0] + 1
        cr = str(rows)

        data = docx2txt.process(filename)
        bulkemails = data

        doc = Doc(bulkemails)
        doc.segment(segmenter)

        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        for span in doc.spans:
            span.normalize(morph_vocab)
        fio = ''
        list_entities = []

        extract_and_store_entities(bulkemails)
        extract_address(bulkemails)

        for span in doc.spans:
            Create_table_entities(span.text, span.type)
            list_entities.append(span.text)
            if span.type == PER:
                span.extract_fact(names_extractor)

    
        names_dict = {_.normal: _.fact.as_dict for _ in doc.spans if _.fact}
        for key in names_dict:
            fio += (key)

        # собираем регулярное выражение в отдельный объект
        r = re.compile('')
        t = re.sub(r'(\b[\w.]+@+[\w.]+.+[\w.]\b)', 'x' + cr, bulkemails)
        t = re.sub(r'(\b\+?[7,8](\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2})\b)', 'x' + cr, t)
        t = re.sub(r'(\d{4}[\s\-]*\d{4}[\s\-]*\d{4}[\s\-]*\d{4})', 'x' + cr, t)
        # дата
        t = re.sub(r'(\d{2}.\d{2}.\d{4}\s\года\s\рождения)', 'x' + cr, t)
        t = re.sub(r'(родившийся\s\d{2}.\d{2}.\d{4}\s\года)', 'x' + cr, t)
        t = re.sub(r'(дата\s\рождения\s\d{2}.\d{2}.\d{4})', 'x' + cr, t)
        # t = re.sub(r'(\d{2}-\d{2}-\d{4})','x'+cr,t)
        # серия паспорта
        t = re.sub(r'(\b\w{5}\b\s\d{4}\b)', 'x' + cr, t)
        # номер паспорта
        t = re.sub(r'(\b\w{5}\b\s\d{6}\b)', 'x' + cr, t)
        # адресу
        t = re.sub(r'(улица\s\w+\b)', 'x' + cr, t)
        t = re.sub(r'(ул.\s\w+\b)', 'x' + cr, t)
        t = re.sub(r'(дом\s\d+\b)', 'x' + cr, t)
        t = re.sub(r'(д.\s\d+\b)', 'x' + cr, t)
        # фио
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я]\.[А-Я]\.)', 'ФИО' + cr, t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+\s)', 'ФИО' + cr + ' ', t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+[,])', 'ФИО' + cr + ' ', t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+[.])', 'ФИО' + cr + '.', t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+[,])', 'ФИО' + cr + ',', t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s)', 'ФИО' + cr + ' ', t)

        person = '30 мая 2023 года,23.05.2022 года,23 мая 2022 года,20.05.2022 года,25.05.2022 года,' \
                 '30 мая 2022 года,19.07.2022 года,04.07.2022,25.05.2022 года,20.05.2022 года,20.05.2022,23.05.2022,' \
                 '25.05.2022 года, 23.01.2022 года, 22.11.2022 года,Пушкино МО,РОССИЙСКОЙ ФЕДЕРАЦИИ,Московской области,' \
                 'ООО «Меркурий МД»,судебной коллегии,Московского областного суда,' \
                 'Пушкинского городского суда,Арбитражный суд,Москвы,Чернозубова О.В.,Крюковой М.В.'
        # person_list = person.split(',')
        # f1_score_result = calculate_f1(person_list, list_entities)
        # st.text(f"F1 Score of anonymization: {f1_score_result:.3f}")
        # calculate_accuracy(person_list, list_entities)
        # st.text(f"accuracy: {f1_score_result:.2f}")

        results = r.findall(bulkemails)
        emails = ""
        for x in results:
            emails += str(x) + "\n"

        # запись результата в файл
        writefile(t)

        st.text_area(label='Вывод начального файла', value=bulkemails)
        con = sqlite3.connect("soob.db")
        cur = con.cursor()
        cur.execute(f"INSERT INTO start (txt) VALUES ('{bulkemails}')")
        con.commit()

        st.text_area(label='Вывод', value=t)
        con = sqlite3.connect("soob.db")
        cur = con.cursor()

        cur.execute(f"INSERT INTO finish (txt) VALUES ('{t}')")
        con.commit()

        st.text_area(label='Вывод', value=str(list_entities).strip('[]'))

###############################################################################################################
elif page == "Работа с текстовым сообщением":
    # Текст
    page_b_img = '''
    <style>
     h4 {
        border: 1px solid red;
        padding: 5px;
        background-color:#545454;
        text-align:center; color: white;

       }

    h3,h5,h6,p {text-align:center; color: white;}

    </style><br><h4>Фабула</h4>
    '''
    st.markdown(page_b_img, unsafe_allow_html=True)
    speech_text = ''

    text = st.text_area('', value=speech_text)
    soob = text

    if Button():

        doc = Doc(soob)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)

        fio = ''
        list_entities = []

        extract_and_store_entities(soob)
        extract_address(soob)

        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        for span in doc.spans:
            span.normalize(morph_vocab)

        for token in doc.tokens:
            token.lemmatize(morph_vocab)

        for span in doc.spans:
            Create_table_entities(span.text, span.type)
            list_entities.append(span.text)
            if span.type == PER:
                span.extract_fact(names_extractor)

        names_dict = {_.normal: _.fact.as_dict for _ in doc.spans if _.fact}
        for key in names_dict:
            fio += (key)

        newfilename = 'result.txt'
        con = sqlite3.connect("soob.db")
        cur = con.cursor()
        cur.execute('SELECT COUNT(*) from start')
        cur_result = cur.fetchone()
        rows = cur_result[0]
        cr = str(rows)

        # почта
        r = re.compile('')
        t = re.sub(r'(\b[\w.]+@+[\w.]+.+[\w.]\b)', 'почта' + cr, soob)
        t = re.sub(r'(\b\+?[7,8](\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2})\b)', 'почта' + cr, t)
        t = re.sub(r'(\d{4}[\s\-]*\d{4}[\s\-]*\d{4}[\s\-]*\d{4})', 'почта' + cr, t)
        # дата
        t = re.sub(r'(\d{2}.\d{2}.\d{4}\s\года\s\рождения)', 'дата' + cr, t)
        t = re.sub(r'(родившийся\s\d{2}.\d{2}.\d{4}\s\года)', 'дата' + cr, t)
        t = re.sub(r'(дата\s\рождения\s\d{2}.\d{2}.\d{4})', 'дата' + cr, t)
        t = re.sub(r'(\d{2}.\d{2}.\d{4}\s\г)', 'дата' + cr, t)
        # t = re.sub(r'(\d{2}-\d{2}-\d{4})','x'+cr,t)
        # серия паспорта
        t = re.sub(r'(\b\w{5}\b\s\d{4}\b)', 'серия' + cr, t)
        # номер паспорта
        t = re.sub(r'(\b\w{5}\b\s\d{6}\b)', 'номер' + cr, t)
        # адресу
        t = re.sub(r'(улица\s\w+\b)', 'ул.' + cr, t)
        t = re.sub(r'(ул.\s\w+\b)', 'д.' + cr, t)
        t = re.sub(r'(дом\s\d+\b)', 'д.' + cr, t)
        t = re.sub(r'(д.\s\d+\b)', 'д.' + cr, t)
        # фио
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я]\.[А-Я]\.)', 'ФИО' + cr, t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+\s)', 'ФИО' + cr + ' ', t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+[,])', 'ФИО' + cr + ' ', t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+[.])', 'ФИО' + cr + '.', t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+[,])', 'ФИО' + cr + ',', t)
        t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s)', 'ФИО' + cr + ' ', t)
        # локация
        # t = re.sub(r'([А-Я][а-я]+\s)', 'Локация' + cr, t)

        person = 'Санкт-Петербург, 25 января 2024 года, Санкт-Петербургском городском суде, Сидорова Андрея Николаевича, ' \
                 'ул. Ленина, д. 123, кв. 45, Юридическая Помощь, пр. Свободы, д. 456, Сидоров Юридическая Помощь, ' \
                 'Сидорова, Сидорова, Сидоровой Ольги Ивановны, Сидорова Ивана Андреевича, Сидоровой Марии Андреевны, ' \
                 'Сидоров, 10 мая 1980 года, Юридическая Помощь, 15 сентября 2010 года, Сидорова, Юридическая Помощь, ' \
                 '15 сентября 2010 года, Сидорова, Юридическая Помощь, Сидорову Андрею Николаевичу, Санкт-Петербург, ' \
                 '25 января 2024 года'
        person_list = person.split(', ')
        # f1_score_result = calculate_f1(person_list, list_entities)
        # st.text(f"F1 Score of anonymization: {f1_score_result:.3f}")
        # calculate_accuracy(person_list, list_entities)
        # st.text(f"accuracy: {f1_score_result:.2f}")

        t = re.sub(fio, 'x' + cr, t)
        con.commit()
        results = r.findall(soob)
        emails = ""
        for x in results:
            emails += str(x) + "\n"

        # запись результата в текстовый файл
        writefile(t)

        st.text_area(label='Вывод', value=t)
        con = sqlite3.connect("soob.db")
        cur = con.cursor()
        cur.execute(f"INSERT INTO finish (txt) VALUES ('{t}')")
        con.commit()

        con = sqlite3.connect("soob.db")
        cur = con.cursor()
        cur.execute(f"INSERT INTO start (txt) VALUES ('{soob}')")
        con.commit()

        st.text_area(label='Вывод', value=str(list_entities).strip('[]'))

        # st.text("--- %s seconds ---" % (time.time() - start_time))

###################################################################################################################################################
elif page == "Работа с файлами PDF":

    try:
        uploaded_file = st.file_uploader('Загрузка файла', type="pdf")
        uploaded_file = uploaded_file.read()

        base64_pdf = base64.b64encode(uploaded_file).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="900" height="1200" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except AttributeError as e:
        pass
    # ПДФ
    if Button():

        # создание базы данных
        newfilename = 'result.txt'
        # создаем соединени с базой данных
        con = sqlite3.connect("soob.db")
        cur = con.cursor()
        #
        cur.execute('SELECT COUNT(*) from start')
        cur_result = cur.fetchone()
        rows = cur_result[0]
        cr = str(rows)

        if uploaded_file:

            uploaded_file = BytesIO(uploaded_file)
            # открываем ПДФ файл для чтения
            pdf_reader = PyPDF2.PdfReader(uploaded_file)

            # создать новый ПДФ файл для записи
            pdf_writer = PyPDF2.PdfWriter()

            # массив для сохранения именованных сущностей
            list_entities = []
            emails = ""
            text1 = ""

            # обход страниц ПДФ файла
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()

                # обработка текста с помощью Наташи
                doc = Doc(page_text)
                doc.segment(segmenter)
                doc.tag_morph(morph_tagger)
                doc.parse_syntax(syntax_parser)
                doc.tag_ner(ner_tagger)

                doc.segment(segmenter)
                doc.tag_morph(morph_tagger)

                extract_and_store_entities(page_text)
                extract_address(page_text)

                for span in doc.spans:
                    span.normalize(morph_vocab)

                for token in doc.tokens:
                    token.lemmatize(morph_vocab)

                # извлечение имён
                fio = ''

                for span in doc.spans:
                    Create_table_entities(span.text, span.type)
                    list_entities.append(span.text)
                    if span.type == PER:
                        span.extract_fact(names_extractor)

                for span in doc.spans:
                    Create_table_entities(span.text, span.type)
                    list_entities.append(span.text)
                    span.extract_fact(dates_extractor)


                names_dict = {_.normal: _.fact.as_dict for _ in doc.spans if _.fact}
                for key in names_dict:
                    fio += (key)

                # почта
                r = re.compile('')
                t = re.sub(r'(\b[\w.]+@+[\w.]+.+[\w.]\b)', 'x' + cr, page_text)
                t = re.sub(r'(\b\+?[7,8](\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2})\b)', 'x' + cr, t)
                t = re.sub(r'(\d{4}[\s\-]*\d{4}[\s\-]*\d{4}[\s\-]*\d{4})', 'x' + cr, t)
                # дата
                t = re.sub(r'(\d{2}.\d{2}.\d{4}\s\года\s\рождения)', 'x' + cr, t)
                t = re.sub(r'(родившийся\s\d{2}.\d{2}.\d{4}\s\года)', 'x' + cr, t)
                t = re.sub(r'(дата\s\рождения\s\d{2}.\d{2}.\d{4})', 'x' + cr, t)
                # t = re.sub(r'(\d{2}-\d{2}-\d{4})','x'+cr,t)
                # серия паспорта
                t = re.sub(r'(\b\w{5}\b\s\d{4}\b)', 'x' + cr, t)
                # номер паспорта
                t = re.sub(r'(\b\w{5}\b\s\d{6}\b)', 'x' + cr, t)
                # адресу
                t = re.sub(r'(улица\s\w+\b)', 'x' + cr, t)
                t = re.sub(r'(ул.\s\w+\b)', 'x' + cr, t)
                t = re.sub(r'(дом\s\d+\b)', 'x' + cr, t)
                t = re.sub(r'(д.\s\d+\b)', 'x' + cr, t)
                # фио
                t = re.sub(r'([А-Я][а-я]+\s+[А-Я]\.[А-Я]\.)', 'ФИО' + cr, t)
                t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+\s)', 'ФИО' + cr + ' ', t)
                t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+[,])', 'ФИО' + cr + ' ', t)
                t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+[.])', 'ФИО' + cr + '.', t)
                t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+[,])', 'ФИО' + cr + ',', t)
                t = re.sub(r'([А-Я][а-я]+\s+[А-Я][а-я]+\s)', 'ФИО' + cr + ' ', t)

                t = re.sub(fio, 'x' + cr, t)
                text1 += t
                con.commit()
                results = r.findall(page_text)

                for x in results:
                    emails += str(x) + "\n"

                person = '30 мая 2023 года,23.05.2022 года,23 мая 2022 года,20.05.2022 года,25.05.2022 года,' \
                         '30 мая 2022 года,19.07.2022 года,04.07.2022,25.05.2022 года,20.05.2022 года,20.05.2022,23.05.2022,' \
                         '25.05.2022 года, 23.01.2022 года, 22.11.2022 года,Пушкино МО,РОССИЙСКОЙ ФЕДЕРАЦИИ,Московской области,' \
                         'ООО «Меркурий МД»,судебной коллегии,Московского областного суда,' \
                         'Пушкинского городского суда,Арбитражный суд,Москвы,Чернозубова О.В.,Крюковой М.В.'
                # person_list = person.split(',')
                # f1_score_result = calculate_f1(person_list, list_entities)
                # st.text(f"F1 Score of anonymization: {f1_score_result:.2f}")
                # calculate_accuracy(person_list, list_entities)
                # st.text(f"accuracy: {f1_score_result:.2f}")

                # Замените имена на "XXXXX" в тексте
                # for name in names_dict.keys():
                #   page_text = page_text.replace(name, "XXXXX")

                # Создайте новую страницу с обновленным текстом
                # new_page = PyPDF2.PageObject.createBlankPage(width=page.mediabox.width(),
                #                                             height=page.mediabox.height())
                # new_page.mergePage(PyPDF2.PageObject.createTextObject(page_text))

                # Добавьте страницу в новый PDF-файл
                pdf_writer.add_blank_page(width=72, height=72)
            # filename = uploaded_file.name
            output_buffer = BytesIO()
            pdf_writer.write(output_buffer)

            # запись результата в текстовый файл
            writefile(text1)

            st.text_area(label='Вывод', value=text1)
            # создаем соединение с базой данных
            con = sqlite3.connect("soob.db")
            cur = con.cursor()
            # добавляем новую запись в таблицу finish
            cur.execute(f"INSERT INTO finish (txt) VALUES ('{text1}')")
            con.commit()

            # st.text_area(label='Вывод', value=page_text)
            con = sqlite3.connect("soob.db")
            cur = con.cursor()
            # добаляем новую запись в таблицу start
            cur.execute(f"INSERT INTO start (txt) VALUES ('{page_text}')")
            con.commit()

            st.text_area(label='Вывод', value=str(', '.join(list_entities)).strip('[]'))

###################################################################################################
elif page == "База данных":
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("https://i.ytimg.com/vi/PgqAvB97Nt0/maxresdefault_live.jpg");
      background-size: cover;
    }

    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    con = sqlite3.connect("soob.db")
    cur = con.cursor()
    file = cur.execute("SELECT * FROM start")
    st.table(file)
    con.commit()
    con = sqlite3.connect("soob.db")
    cur = con.cursor()
    fil = cur.execute("SELECT * FROM finish")
    st.table(fil)
    con.commit()
    con = sqlite3.connect("soob.db")
    cur = con.cursor()
    fill = cur.execute("SELECT * FROM named_entities")
    st.table(fill)
    con.commit()
    con.close()

    # кнопка для удаления
    my_button = st.markdown("""
        <style >
        div.stButton > button: {
            position: absolute;
        }
        div.stButton > button:first-child {
            background-color: #578a00;
            color:#ffffff;
        }
        div.stButton > button:hover {
            background-color: #00128a;
            color:#ffffff;
            }
        </style>""", unsafe_allow_html=True)

    my_button = st.button('Очистить базу')
    if my_button:
        con = sqlite3.connect("soob.db")
        cur = con.cursor()
        delete_file = cur.execute('DELETE FROM start')
        delete_file = cur.execute('DELETE FROM finish')
        delete_file = cur.execute('DELETE FROM named_entities')
        con.commit()
        con.close()