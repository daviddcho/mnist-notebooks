def fetch(url):
  import requests, os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp) and os.stat(fp).st_size > 0 and os.getenv("NOCACHE", None) is None:
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    print("fetching %s" % url)
    r = requests.get(url)
    assert r.status_code == 200
    dat = r.content
    with open(fp+".tmp", "wb") as f:
      f.write(dat)
    os.rename(fp+".tmp", fp)
  return dat
