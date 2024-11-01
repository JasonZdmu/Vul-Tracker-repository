# -*- coding:utf-8 -*-
 
import requests
import re
from pyDes import des, PAD_PKCS5, ECB
import binascii
import time
 
 
def des_encrypt(s, key):
    """
    DES 加密
    :param key: 秘钥
    :param s: 原始字符串
    :return: 加密后字符串，16进制
    """
    secret_key = key
    k = des(secret_key, mode=ECB, pad=None, padmode=PAD_PKCS5)
    en = k.encrypt(s, padmode=PAD_PKCS5)
    return en  # 得到加密后的16位进制密码 <class 'bytes'>
 
 
def encrypt(pd='12345', key='aM51f8FuE/s='):
    """
    密码加密过程：
    1 从认证页面中可获得base64格式的秘钥
    2 将秘钥解码成bytes格式
    3 输入明文密码
    4 通过des加密明文密码
    5 返回base64编码格式的加密后密码
    :param pd: 明文密码
    :param key: 秘钥
    :return: 加密后的密码（base64格式）
    """
    key = binascii.a2b_base64(key.encode('utf-8'))  # 先解码 <class 'bytes'>
    pd_bytes = des_encrypt(pd, key)  # 得到加密后的16位进制密码 <class 'bytes'>
    pd_base64 = binascii.b2a_base64(pd_bytes, newline=False).decode('utf-8')
    return pd_base64
 
 
def login(username, password):
    start_time = time.process_time()
    session = requests.session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46'
    }
    session.headers = headers
 
    # 访问任意网址，返回包含认证页面链接的内容（自动跳转）
    url = 'http://baidu.com/'
    resp = session.get(url, verify=False)
 
    # 提取认证链接并访问，经历一次重定向得到认证页面，且会返回一个session值
    url = re.search(r"href='(.*?)'</script>", resp.text).group(1)
    resp = session.get(url)
 
    # '''从认证页面正则得到 croypto（密钥 base64格式） 与 execution（post参数）的值 '''
    croypto = re.search(r'"login-croypto">(.*?)<', resp.text, re.S).group(1)
    execution = re.search(r'"login-page-flowkey">(.*?)<', resp.text, re.S).group(1)
    # 构建post数据 填入自己的学号 密码
    data = {
        'username': username,  # 学号
        'type': 'UsernamePassword',
        '_eventId': 'submit',
        'geolocation': '',
        'execution': execution,
        'captcha_code': '',
        'croypto': croypto,  # 密钥 base64格式
        'password': encrypt(password, croypto)  # 密码 经过des加密 base64格式
    }
 
    # 添加cookie值
    session.cookies.update({'isPortal': 'false'})
 
    # 提交数据，进行登录，这里禁止重定向，因为会有cookie限制
    url = 'https://id.dlmu.edu.cn/login'
    resp = session.post(url, data=data, allow_redirects=False)
 
    # 得到上一步返回的重定向网址，继续访问（需要清空cookie值）
    # 这里实际经过了三次重定向
    url = resp.headers['Location']
    session.cookies.clear()
    resp = session.get(url)
 
    end_time = time.process_time()
    print(end_time - start_time)
    if resp.status_code == 200:
        print('成功登录')
 
 
if __name__ == '__main__':
    username = '1120231545'
    password = 'ZJzj&1234'
    login(username, password)
