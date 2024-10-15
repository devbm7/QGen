import pandas as pd
from fpdf import FPDF
import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText 
# from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
# from email import encoders


def export_to_csv(data):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    return csv

def export_to_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for item in data:
        pdf.multi_cell(0, 10, f"Context: {item['context']}")
        pdf.multi_cell(0, 10, f"Question: {item['question']}")
        pdf.multi_cell(0, 10, f"Answer: {item['answer']}")
        pdf.multi_cell(0, 10, f"Options: {', '.join(item['options'])}")
        pdf.multi_cell(0, 10, f"Overall Score: {item['overall_score']:.2f}")
        pdf.ln(10)
    
    return pdf.output(dest='S').encode('latin-1')

def send_email_with_attachment(email_subject, email_body, recipient_emails, sender_email, sender_password, attachment):
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server
    smtp_port = 587  # Replace with your SMTP port

    # Create the email message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = ", ".join(recipient_emails) 
    message['Subject'] = email_subject
    message.attach(MIMEText(email_body, 'plain'))

    # Attach the feedback data if available
    if attachment:
        attachment_part = MIMEApplication(attachment.getvalue(), Name="feedback_data.json")
        attachment_part['Content-Disposition'] = f'attachment; filename="feedback_data.json"'
        message.attach(attachment_part)

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            # print(sender_email)
            # print(sender_password)
            server.login(sender_email, sender_password)
            text = message.as_string()
            server.sendmail(sender_email, recipient_emails, text)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False
