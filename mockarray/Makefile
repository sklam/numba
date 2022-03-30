libextarray.so: extarray.c
	$(CC) -I.. -shared -o libextarray.so extarray.c


test: libextarray.so
	pytest extarray.py

all: test


clean:
	rm libextarray.so