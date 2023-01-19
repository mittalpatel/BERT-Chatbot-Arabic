# ==============================================================================
# title              : app.py
# description        : This is the flask app for Bert closed domain chatbot which accepts the user request and response back with the answer
# author             : Pragnakalp Techlabs
# email              : letstalk@pragnakalp.com
# website            : https://www.pragnakalp.com
# python_version     : 3.6.x +
# ==============================================================================

# Import required libraries
from flask import Flask, render_template, request
from flask_cors import CORS
import email
import csv
import datetime
import smtplib
import ssl
import socket
from email.mime.text import MIMEText
from bert import QA

timestamp = datetime.datetime.now()
date = timestamp.strftime('%d-%m-%Y')
time = timestamp.strftime('%I:%M:%S')
IP = ''

app = Flask(__name__)
CORS(app)

# Provide the fine_tuned model path in QA Class
model_ar = QA("arabic_model_bin")

# This is used to show the home page
@app.route("/")
def home():
    return render_template("home.html")

# This is used to give response 
@app.route("/predict")
def get_bot_response():   
    IP = request.remote_addr
    q = request.args.get('msg')
    bert_bot_log = []
    bert_bot_log.append(q)
    bert_bot_log.append(date)
    bert_bot_log.append(time)
    bert_bot_log.append(IP)
    
    # You can provide your own paragraph from here
    arabic_para = "يضاف إلى ذلك توفيرها لإمكانية نشر المواقع التي توفر معلومات نصية ورسومية في شكل قواعد بيانات وخرائط على شبكة الإنترنت وبرامج الأوفيس وإتاحة أوركوت التي تتيح الاتصال عبر الشبكة بين الأفراد ومشاركة أفلام وعروض الفيديو، علاوةً على الإعلان عن نسخ مجانية إعلانية من الخدمات التكنولوجية السابقة. يقع المقرّ الرئيسي للشركة، والذي يحمل اسم جوجل بليكس، في مدينة ماونتن فيو بولاية كاليفورنيا. وقد وصل عدد موظفيها الذين يعملون دوامًا كاملًا في 31 مارس عام 2009 إلى 20,164 موظفًا. تأسست هذه الشركة على يد كل من لاري بايج وسيرجي برين عندما كانا طالبين بجامعة ستانفورد. في بادئ الأمر تم تأسيس الشركة في الرابع من سبتمبر عام 1998 كشركة خاصة مملوكة لعدد قليل من الأشخاص. وفي التاسع عشر من أغسطس عام 2004، طرحت الشركة أسهمها في اكتتاب عام ابتدائي، لتجمع الشركة بعده رأس مال بلغت قيمته 1.67 مليار دولار أمريكي، وبهذه القيمة وصلت قيمة رأس مال الشركة بأكملها إلى 23 مليار دولار أمريكي. وبعد ذلك واصلت شركة جوجل ازدهارها عبر طرحها لسلسلة من المنتجات الجديدة واستحواذها على شركات أخرى عديدة والدخول في شراكات جديدة. وطوال مراحل ازدهار الشركة، كانت ركائزها المهمة هي المحافظة على البيئة وخدمة المجتمع والإبقاء على العلاقات الإيجابية بين موظفيها. ولأكثر من مرة، احتلت الشركة المرتبة الأولى في تقييم لأفضل الشركات تجريه مجلة فورتشن كما حازت بصفة أقوى مئة علامة تجارية في العالم الذي تجريه مجموعة شركات ميلوارد براون."

    # This function creates a log file which contain the question, answer, date, time, IP addr of the user
    def bert_log_fn(answer_err):
        bert_bot_log.append(answer_err)
        with open('bert_bot_log.csv', 'a', encoding='utf-8') as logs:
            write = csv.writer(logs)
            write.writerow(bert_bot_log)
        logs.close()


    # This block calls the prediction function and return the response
    try:        
        out = model_ar.predict(arabic_para, q)
        confidence = out["confidence"]
        confidence_score = round(confidence*100)
        if confidence_score > 10:
            bert_log_fn(out["answer"])
            return out["answer"]
        else:
            bert_log_fn("Sorry I don't know the answer, please try some different question.")
            return "Sorry I don't know the answer, please try some different question."         
    except Exception as e:
        bert_log_fn("Sorry, Server doesn't respond..!!")
        print("Exception Message ==> ",e)
        return "Sorry, Server doesn't respond..!!"

# You can change the Flask app port number from here.
if __name__ == "__main__":
    app.run(port='3000')
