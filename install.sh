#!/bin/sh

install -dm 755 "/usr/bin"
install -dm 755 "/usr/share/passgan"

install -Dm 644 -t "/usr/share/doc/passgan/" *.md

install -Dm 644 LICENSE "/usr/share/licenses/passgan/LICENSE"

rm -rf *.md LICENSE

cp -a --no-preserve=ownership * "/usr/share/passgan/"

cat > "/usr/bin/passgan" << EOF
#!/bin/sh

exec python /usr/share/passgan/passgan.py "\$@"
EOF

chmod a+x "/usr/bin/passgan"
