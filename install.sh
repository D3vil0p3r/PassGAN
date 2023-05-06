#!/bin/sh

install -dm 755 "/usr/bin"
install -dm 755 "/usr/share/passgan"

install -Dm 644 -t "/usr/share/doc/passgan/" README.md 2>/dev/null

install -Dm 644 LICENSE "/usr/share/licenses/passgan/LICENSE" 2>/dev/null

rm -rf README.md LICENSE

cp -a --no-preserve=ownership * "/usr/share/passgan/"

cat > "/usr/bin/passgan" << EOF
#!/bin/sh

exec python /usr/share/passgan/passgan.py "\$@"
EOF

chmod a+x "/usr/bin/passgan"
echo "PassGAN installed."
