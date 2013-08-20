# Maintainer: Matthew Wardrop <mister.wardrop@gmail.com>
pkgname=python2-hdf5storage
pkgver=0.1
pkgrel=1
pkgdesc="A storage container for data (with attributes) that can be persisted as HDF5 files using pytables."
arch=('i686' 'x86_64')
url=""
license=('GPL')
groups=()
depends=('python2' 'python2-numpy' 'python2-pytables' 'python2-scipy')
makedepends=()
provides=()
conflicts=()
replaces=()
backup=()
options=(!emptydirs)
install=
source=()
md5sums=()

package() {
  cd ".."
  #cd "$srcdir/$pkgname-$pkgver"
  python2 setup.py install --root="$pkgdir/" --optimize=1
}

# vim:set ts=2 sw=2 et:
