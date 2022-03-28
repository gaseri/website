---
author: Vedran Miletić
---

# Mjerenje performansi web poslužitelja alatom ab (Apache HTTP server benchmarking tool)

Apache HTTP Server uključuje ApacheBench, kraće ab, koji služi za mjerenje performansi web poslužitelja. Specijalno, unutar slike `httpd` na Docker Hubu, ab je dostupan kao naredba `ab` koju je moguće pokrenuti navođenjem te naredbe nakon imena i verzije slike:

``` shell
$ docker run httpd:2.4 ab
ab: wrong number of arguments
Usage: ab [options] [http[s]://]hostname[:port]/path
Options are:
    -n requests     Number of requests to perform
    -c concurrency  Number of multiple requests to make at a time
    -t timelimit    Seconds to max. to spend on benchmarking
                    This implies -n 50000
    -s timeout      Seconds to max. wait for each response
                    Default is 30 seconds
    -b windowsize   Size of TCP send/receive buffer, in bytes
    -B address      Address to bind to when making outgoing connections
    -p postfile     File containing data to POST. Remember also to set -T
    -u putfile      File containing data to PUT. Remember also to set -T
    -T content-type Content-type header to use for POST/PUT data, eg.
                    'application/x-www-form-urlencoded'
                    Default is 'text/plain'
    -v verbosity    How much troubleshooting info to print
    -w              Print out results in HTML tables
    -i              Use HEAD instead of GET
    -x attributes   String to insert as table attributes
    -y attributes   String to insert as tr attributes
    -z attributes   String to insert as td or th attributes
    -C attribute    Add cookie, eg. 'Apache=1234'. (repeatable)
    -H attribute    Add Arbitrary header line, eg. 'Accept-Encoding: gzip'
                    Inserted after all normal header lines. (repeatable)
    -A attribute    Add Basic WWW Authentication, the attributes
                    are a colon separated username and password.
    -P attribute    Add Basic Proxy Authentication, the attributes
                    are a colon separated username and password.
    -X proxy:port   Proxyserver and port number to use
    -V              Print version number and exit
    -k              Use HTTP KeepAlive feature
    -d              Do not show percentiles served table.
    -S              Do not show confidence estimators and warnings.
    -q              Do not show progress when doing more than 150 requests
    -l              Accept variable document length (use this for dynamic pages)
    -g filename     Output collected data to gnuplot format file.
    -e filename     Output CSV file with percentages served
    -r              Don't exit on socket receive errors.
    -m method       Method name
    -h              Display usage information (this message)
    -I              Disable TLS Server Name Indication (SNI) extension
    -Z ciphersuite  Specify SSL/TLS cipher suite (See openssl ciphers)
    -f protocol     Specify SSL/TLS protocol
                    (SSL2, TLS1, TLS1.1, TLS1.2 or ALL)
    -E certfile     Specify optional client certificate chain and private key
```

[Službena dokumentacija ab-a](https://httpd.apache.org/docs/2.4/programs/ab.html) opisuje sve dostupne parametre i njihove uloge, a osnovni način korištenja je uz navođenje parametara `-n` s ukupnim brojem zahjteva koji će se izvesti i `-c` s ukupnim brojem istovremenih zahjeva:

``` shell
$ docker run httpd:2.4 ab -n 1000 -c 100 http://172.17.0.2/
This is ApacheBench, Version 2.3 <$Revision: 1879490 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 172.17.0.2 (be patient)
Completed 100 requests
Completed 200 requests
Completed 300 requests
Completed 400 requests
Completed 500 requests
Completed 600 requests
Completed 700 requests
Completed 800 requests
Completed 900 requests
Completed 1000 requests
Finished 1000 requests


Server Software:        Apache/2.4.53
Server Hostname:        172.17.0.2
Server Port:            80

Document Path:          /
Document Length:        45 bytes

Concurrency Level:      100
Time taken for tests:   0.079 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      289000 bytes
HTML transferred:       45000 bytes
Requests per second:    12689.87 [#/sec] (mean)
Time per request:       7.880 [ms] (mean)
Time per request:       0.079 [ms] (mean, across all concurrent requests)
Transfer rate:          3581.42 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    3   0.6      3       5
Processing:     1    4   0.7      4       7
Waiting:        0    3   0.9      3       5
Total:          3    7   0.5      7      10

Percentage of the requests served within a certain time (ms)
  50%      7
  66%      8
  75%      8
  80%      8
  90%      8
  95%      8
  98%      8
  99%      9
 100%     10 (longest request)
```
