from quafu import User

def save_apitoken():
    user = User()
    user.save_apitoken(input("Please input your API Token:\n"))