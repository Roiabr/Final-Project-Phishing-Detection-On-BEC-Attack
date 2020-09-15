class Email:
    def __init__(self, count, header, body, label):
        self.count = count
        self.header = header
        self.body = body
        self.label = label

    @classmethod
    def List_of_emails(cls, data):
        count = 0
        list_email = []
        for row in data.values:
            email = Email(count, row[0], row[1], row[2])
            if email.header == 'NaN' or email.body == 'NaN':
                pass
            count = count + 1
            list_email.append(email)
        return list_email
