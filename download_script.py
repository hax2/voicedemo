import urllib.request
url = 'https://raw.githubusercontent.com/Plachtaa/seed-vc/main/modules/v2/vc_wrapper.py'
with urllib.request.urlopen(url) as response:
    content = response.read().decode('utf-8')
    with open('c:/Users/profe/voice/fetched_vc_wrapper.py', 'w', encoding='utf-8') as f:
        f.write(content)
print(f"Downloaded {len(content)} bytes.")
