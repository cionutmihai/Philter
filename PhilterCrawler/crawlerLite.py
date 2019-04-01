
import shodan
import sys

API_KEY = "nice try :)"

if len(sys.argv) == 1:
        print ('Usage: %s <search query>' % sys.argv[0])
        sys.exit(1)

try:

        api = shodan.Shodan(API_KEY)

        query = ' '.join(sys.argv[1:])
        result = api.search(query)

        with open('Results.txt', 'w') as f:
            for service in result['matches']:
                    print (service['ip_str'])
                    f.write("%s\n" % service['ip_str'])

except Exception as e:
        print ('Error: %s' % e)
        sys.exit(1)

