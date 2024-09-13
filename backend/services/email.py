import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email(sender_email, receiver_email, subject, body, password):
    # 创建MIMEMultipart对象来表示邮件
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # 添加邮件正文
    msg.attach(MIMEText(body, 'plain'))

    try:
        # 连接到SMTP服务器
        print("正在连接到SMTP服务器...")
        server = smtplib.SMTP('smtp.buaa.edu.cn', 587)
        server.starttls()  # 启用TLS加密
        print("成功连接到服务器，正在登录...")

        # 登录邮箱
        server.login(sender_email, password)
        print("登录成功，正在发送邮件...")

        # 发送邮件
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("邮件发送成功")

    except Exception as e:
        print(f"邮件发送失败: {e}")
    finally:
        server.quit()
        print("已断开服务器连接")


# 使用示例
sender_email = "j.x.wang@buaa.edu.cn"
receiver_email = "2474562672@qq.com"
subject = "测试邮件"
body = "你好，这是一个通过Python发送的测试邮件。"
password = ""  #
send_email(sender_email, receiver_email, subject, body, password)
