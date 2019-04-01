from __future__ import print_function
import argparse
import csv
import json
import sys
import os
import unix_converter as unix
from urllib.request import urlopen


def main(address, output_dir):
    raw_account = get_address(address)
    account = json.loads(raw_account.read())
    print_header(account)
    parse_transactions(account, output_dir)


def get_address(address):
    url = 'https://blockchain.info/address/{}?format=json'
    formatted_url = url.format(address)
    return urlopen(formatted_url)


def parse_transactions(account, output_dir):
    transactions = []
    for i, tx in enumerate(account['txs']):
        transaction = []
        outputs = {}
        inputs = get_inputs(tx)
        transaction.append(i)
        transaction.append(unix.unix_converter(tx['time']))
        transaction.append(tx['hash'])
        transaction.append(inputs)
        for output in tx['out']:
            outputs[output['addr']] = output['value'] * 10 ** -8
        transaction.append('\n'.join(outputs.keys()))
        transaction.append(
            '\n'.join(str(v) for v in outputs.values()))
        transaction.append('{:.8f}'.format(sum(outputs.values())))
        transactions.append(transaction)
    csv_writer(transactions, output_dir)


def print_header(account):
    print('Address:', account['address'])
    print('Current Balance: {:.8f} BTC'.format(account['final_balance'] * 10 ** -8))
    print('Total Sent: {:.8f} BTC'.format(account['total_sent'] * 10 ** -8))
    print('Total Received: {:.8f} BTC'.format(account['total_received'] * 10 ** -8))
    print('Number of Transactions:', account['n_tx'])

    print('{:=^22}\n'.format(''))


def get_inputs(tx):
    inputs = []
    for input_addr in tx['inputs']:
        inputs.append(input_addr['prev_out']['addr'])
    if len(inputs) > 1:
        input_string = '\n'.join(inputs)
    else:
        input_string = ''.join(inputs)
    return input_string


def csv_writer(data, output_dir):
    print('Writing output.')
    headers = ['Index', 'Date', 'Transaction Hash', 'Inputs', 'Outputs', 'Values', 'Total']
    try:
        if sys.version_info[0] == 2:
            csvfile = open(output_dir, 'wb')
        else:
            csvfile = open(output_dir, 'w', newline='')
        with csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for transaction in data:
                writer.writerow(transaction)
            csvfile.flush()
            csvfile.close()
    except IOError as e:
        print("""Error writing to CSV file.
		Please check output argument {}""".format(e.filename))
        sys.exit(1)
    print('Program closing.')
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Philter BTC Address Forensics', epilog='Have fun')
    parser.add_argument('ADDR', help='Bitcoin Address')
    parser.add_argument('OUTPUT', help='Output CSV file')
    args = parser.parse_args()
    print('{:=^22}'.format(''))
    print('{}'.format('Philter Bitcoin Address Forensics'))
    print('{:=^22} \n'.format(''))
    main(args.ADDR, args.OUTPUT)
