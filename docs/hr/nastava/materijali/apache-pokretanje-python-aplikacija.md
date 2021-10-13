---
author: Vedran Miletić
---

# Pokretanje Python web aplikacija u web poslužitelju Apache HTTP Server

- modul za Web Server Gateway Interface (WSGI), najpopularniji način korištenja Python web aplikacija
- paket se zove `mod_wsgi`
- jednostavna WSGI aplikacija ima kod oblika

    ``` python
    def application(environ, start_response):
        status = '200 OK'
        output = 'Ovo je demo WSGI aplikacija.'

        response_headers = [('Content-Type', 'text/plain'),
                            ('Content-Length', str(len(output)))]
        start_response(status, response_headers)

        return [output]
    ```

- uključivanje WSGI skripte u web server

    ``` apacheconf
    WSGIScriptAlias /myapp /usr/local/www/wsgi-scripts/myapp.wsgi
    ```

- konfiguracija dozvola

    ``` apacheconf
    <Directory /var/www/wsgi-scripts>
      Order allow,deny
      Allow from all
    </Directory>
    ```
