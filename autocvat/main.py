from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from tqdm import tqdm
from selenium.webdriver.common.keys import Keys
import time
import cv2
from shutil import rmtree
service = Service('/home/jy/projects/autocvat/chromedriver')
service.start()
driver = webdriver.Remote(service.service_url)
driver.get('http://192.168.100.222:8080/auth/login')
username = driver.find_element_by_id("username")
password = driver.find_element_by_id("password")

username.send_keys("ilab")
password.send_keys("ilab")
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="root"]/div/div/form/div[3]/div/div/div/button').click()

done = ['D81141255','E57379120','C90842467','D81142586', 'C90843930', 'D88628921'
        'D81141708',
'E57379538',
'D00269443',
'D81141274',
'E57381711',
'E57379241',
'E57378873',
'D81141462',
'C90843485',
'D88629336',
'E57378949',
'E57379387',
'D81146705',
'E82842693',
'D00269541',
'E57378832',
'C90843200',
'C90842488',
'D81139315',
'E57379558',
'D00267903',
'E57379590',
'D81141623',
'C90843479',
'D81141444',
'D88629181',
'D22710478',
'D81141460',
'D88628609',
'D00268651',
'E82843260',
'D81139831',
'E58579194',
'C90842487',
'C90843803',
'D81141170',
'E57378959',
'D81141562',
'D81139848',
'D00268624',
'D81143244',
'D81146550',
'E66668353',
'D70007834',
'D00269554',
'E57379415',
'C90840508',
'E66667878',
'D00269480',
'D88629116',
'D81139221',
'D81141630',
'D81142870',
'E57378847',
'C90843589'
        ]
# with open('2021-02-03-14-23-52_EXPORT_CSV_2088113_040_0.csv','r')as txt:
#     filelist = txt.readlines()
for root,dirs,files in os.walk('videosultra_1'):
    for d in tqdm(dirs):
        if  d not in done:
            print(d)

            filess = os.listdir(os.path.join(root,d))
            smallest_file = ""
            smallest_file_size = 100000
            f = filess[0]
            # for f in filess:
            #
            #     f = os.path.join(root,d,f)
            #     if os.path.isfile(f):
            #         size = os.path.getsize(f)
            #         if size < smallest_file_size:
            #             smallest_file_size = size
            #             smallest_file = f
            filename = f.split('/')[-1]
            f = os.path.join(root, d, f)

            cap = cv2.VideoCapture(f)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fps = int(fps/2)

            # for ftxt in filelist:
            #     if filename in ftxt:
            #         ftxt = ftxt.replace('"','')
            #         url = ftxt.split(',')[2]
            #         print(url)
                    #
            time.sleep(1)
            driver.get('http://192.168.100.222:8080/tasks')
            time.sleep(3)
            try:
                driver.find_element_by_xpath('//*[@id="cvat-create-task-button"]').click()
            except Exception:
                continue

            time.sleep(0.5)


            driver.find_element_by_xpath('//*[@id="name"]').send_keys(filename.split('.')[0])
            # driver.find_element_by_xpath('// *[ @ id = "rc_select_0"]').send_keys('Sink_only')
            time.sleep(0.5)
            # wait = WebDriverWait(driver, 60)
            driver.find_element_by_xpath('// *[ @ id = "rc_select_0"]').click()
            # female = wait.until(
            #     EC.visibility_of_element_located((By.CSS_SELECTOR, '.Select-option#rc_select_0_list_0)')))
            #
            # # Click the element female option value of the dropdown
            # female.click()
            time.sleep(1)
            try:
                driver.find_element_by_xpath('/html/body/div[3]/div/div/div/div[2]/div[1]/div/div/div/div').click()
            except Exception:
                driver.find_element_by_xpath('/html/body/div[4]/div/div/div/div[2]/div[1]/div/div/div[1]/div').click()
            time.sleep(0.5)
            # driver.find_element_by_xpath('// *[ @ id = "rc-tabs-1-tab-remote"]').click()

            # driver.find_element_by_xpath('//*[@id="rc-tabs-1-panel-remote"]/textarea').send_keys(url)
            print(f)
            driver.find_element_by_xpath('//*[@id="rc-tabs-1-panel-local"]/span/div[1]/span/input').send_keys(f)
            time.sleep(0.5)
            driver.find_element_by_xpath('//*[@id="root"]/section/main/div/div/div/div[8]/div/div/div/span[2]').click()
            time.sleep(0.5)
            driver.find_element_by_xpath('// *[ @ id = "frameStep"]').send_keys(str(fps))

            time.sleep(0.5)
            driver.find_element_by_xpath('//*[@id="imageQuality"]').send_keys(Keys.BACK_SPACE)
            driver.find_element_by_xpath('//*[@id="imageQuality"]').send_keys(Keys.BACK_SPACE)
            driver.find_element_by_xpath('//*[@id="imageQuality"]').send_keys(str(100))
            time.sleep(0.5)
            driver.find_element_by_xpath('//*[@id="root"]/section/main/div/div/div/div[10]/button/span').click()
            time.sleep(1)
            try:
                element = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="root"]/section/main/div/div/div/div[9]/button'))
                )
            except Exception:
                pass
            rmtree(os.path.join(root,d))




