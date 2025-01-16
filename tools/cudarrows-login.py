from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import argparse

parser = argparse.ArgumentParser(
                    prog="CUDArrows Login Manager",
                    description="Setups authorization cookie for CUDArrows")

parser.add_argument("--browser", choices=["firefox", "chrome"], default="chrome")

args = parser.parse_args()

OAUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

CLIENT_ID = "480103000605-r44ki2s4brehsbih1nb676gnialp4a5v.apps.googleusercontent.com"

port = 3000

params = "?" \
    "redirect_uri=https://logic-arrows.io/api/auth/google&" \
    "client_id=" + CLIENT_ID + "&" \
    "access_type=offline&" \
    "response_type=code&" \
    "prompt=consent&" \
    "scope=https://www.googleapis.com/auth/userinfo.email"

url = f"{OAUTH_URL}{params}"

match args.browser:
    case "firefox":
        profile = webdriver.FirefoxProfile()
        profile.set_preference("dom.webdriver.enabled", False)
        profile.set_preference("useAutomationExtension", False)
        profile.update_preferences()
        driver = webdriver.Firefox(firefox_profile=profile)
    case "chrome":
        opts = Options()
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_argument("disable-blink-features=AutomationControlled")
        driver = webdriver.Chrome(opts)

driver.get(url)

WebDriverWait(driver, 3600).until(EC.url_to_be("https://logic-arrows.io/maps"))

user_agent = driver.execute_script("return navigator.userAgent;")

token = driver.get_cookie("accessToken")["value"]

with open("session.txt", "w") as file:
    file.write(f"{user_agent}\n{token}")

driver.close()