import os
from secrets import token_hex
class Utility:
    def __init__(self):
        self.allowed_type = ['csv', 'json', 'xlsx', 'txt']

    def save_file(self, file_content, filename, folder):
        file_ext = filename.split(".").pop()
        # Check if filetype is allowed
        if(file_ext in self.allowed_type):
            # Check if folder exists
            if(os.path.exists(folder)):
                # Check if file exists in folder
                if(os.path.exists(f"{folder}/{filename}")):
                    # Change filename
                    new_filename = f"{filename.split('.')[0]}{token_hex(5)}.{file_ext}"
                    # Save file
                    with open(f"{folder}/{new_filename}", 'wb') as f:
                        f.write(file_content)
                    return {"saved":True, "filename":new_filename}
                else:
                    # Save file
                    with open(f"{folder}/{filename}", 'wb') as f:
                        f.write(file_content)
                    return {"saved":True, "filename":filename}
            else:
                # Create folder
                os.mkdir(folder)
                # Check if New folder exists (Permission reasons)
                if(os.path.exists(folder)):
                    # Save file
                    with open(f"{folder}/{filename}", 'wb') as f:
                        f.write(file_content)
                    return {"saved":True, "filename":filename}
                return {"saved":False, "filename":filename}

        return {"saved":False, "filename":filename}
    
    def read_dataset(self, filename, folder='datasets'):
        file_ext = filename.split(".").pop()

        if(file_ext in self.allowed_type):
            if(f"{folder}/{filename}"):
                if(file_ext == 'csv'):
                    df = pd.read_csv(f"{folder}/{filename}")
                    return {"read": True, "dataframe":df}
                
                if(file_ext == 'json'):
                    df1 = pd.read_json(f"{folder}/{filename}")
                    dfc = df1.to_csv(f"{folder}/{filename.split('.')[0]}.csv")
                    df = df = pd.read_csv(f"{folder}/{filename}")
                    return {"read": True, "dataframe":df}
                
                if(file_ext == 'txt'):
                    df = pd.read_csv(f"{folder}/{filename}", sep=" ", header=None)
                    return {"read": True, "dataframe":df}
                    
            return {"read": False, "Message": "File does not exist / Unable to read file "}
        return {"read": False, "Message": "Invalid / maliciious file"}

