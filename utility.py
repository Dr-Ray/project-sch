import os
from secrets import token_hex
class Utility:
    def __init__(self):
        self.allowed_type = ['csv', 'json', 'xlsx', 'txt']

    def save_file(self, file_content, filename, folder):
        file_ext = filename.split(".").pop()
        
        if(file_ext in self.allowed_type):
            if(os.path.exists(folder)):
                if(f"{folder}/{filename}"):
                    new_filename = f"{filename.split('.')[0]}{token_hex(5)}.{file_ext}"
                    with open(f"{folder}/{new_filename}", 'wb') as f:
                        f.write(file_content)
                        return True
                else:
                    with open(f"{folder}/{filename}", 'wb') as f:
                        f.write(file_content)
                        return True
            else:
                os.mkdir(folder)
                if(os.path.exists(folder)):
                    with open(f"{folder}/{filename}", 'wb') as f:
                        f.write(file_content)
                        return True
                return False

        return False
