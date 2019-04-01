
import datetime

def main():
    unix_ts = int(raw_input('Unix timestamp to convert:\n>> '))
    print (unix_converter(unix_ts))

def unix_converter(timestamp):
    date_ts = datetime.datetime.utcfromtimestamp(timestamp)
    return date_ts.strftime('%m/%d/%Y %I:%M:%S %p UTC')

if __name__ == '__main__':

	main()

