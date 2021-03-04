from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from tqdm import tqdm
import time
from shutil import rmtree
from selenium.webdriver.common.keys import Keys


service = Service('/home/jy/projects/workingscripts/autocvat/chromedriver')
service.start()
driver = webdriver.Remote(service.service_url)
driver.get('http://192.168.100.222:8080/auth/login')
time.sleep(0.5)
username = driver.find_element_by_id("username")
password = driver.find_element_by_id("password")

username.send_keys("ilab")
password.send_keys("ilab")
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="root"]/div/div/form/div[3]/div/div/div/button').click()
time.sleep(1)
driver.get('http://192.168.100.222:8080/projects/6')
time.sleep(15)


t = 3
i=275
while True:

    try:
        time.sleep(0.5)
        dropdownbtn = driver.find_element_by_css_selector('#root > section > main > div > div > div:nth-child('+str(i)+') > div:nth-child(4) > div:nth-child(2) > div > span.anticon.ant-dropdown-trigger.cvat-menu-icon > svg')
        dropdownbtn.click()
        time.sleep(0.5)

        driver.find_element_by_xpath('/html/body/div[' + str(t) + ']/div/div/ul/li[3]/div').click()

        time.sleep(0.5)
        driver.find_element_by_xpath('/html/body/div[' + str(t + 1) + ']/div/div/ul/li[13]').click()
        time.sleep(0.5)
        t += 2
        i+=1
    except Exception:
        print(Exception)
        pass
#
# for i in range(40,46):
#     driver.find_element_by_css_selector('#root > section > main > div > div:nth-child(3) > div > ul > li.ant-pagination-options > div > input[type=text]').send_keys(str(i))
#                                           # root > section > main > div > div:nth-child(3) > div > ul > li.ant-pagination-options > div > input[type=text]
#     time.sleep(0.5)
#     driver.find_element_by_class_name('ant-pagination-item-link').click()
#     time.sleep(1)
#     t = 3
#     for i in range(1,10):
#         containers = driver.find_elements_by_css_selector(
#             '#root > section > main > div > div:nth-child(2) > div > div:nth-child('+ str(i) +')')
#
#         for item in containers:
#
#             dropdownbtn = item.find_element_by_css_selector('div:nth-child('+ str(i) +') > div:nth-child(4) > div:nth-child(2) > div > span.anticon.ant-dropdown-trigger.cvat-menu-icon')
#             dropdownbtn.click()
#             time.sleep(0.5)
#
#             driver.find_element_by_xpath('/html/body/div['+str(t)+']/div/div/ul/li[3]/div').click()
#
#             time.sleep(0.5)
#             driver.find_element_by_xpath('/html/body/div['+str(t+1)+']/div/div/ul/li[13]').click()
#             time.sleep(0.5)
#             t+=2
time.sleep(1000)
