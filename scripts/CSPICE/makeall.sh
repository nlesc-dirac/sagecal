echo This script builds the SPICE delivery
echo for the cspice package of the toolkit.
echo
echo The script must be executed from the
echo cspice directory.
echo
cd src
echo
echo Creating cspice
echo
cd cspice
../../mkprodct.sh
cd ..
echo
echo Creating csupport
echo
cd csupport
../../mkprodct.sh
cd ..
echo
echo Creating brief_c
echo
cd brief_c
../../mkprodct.sh
cd ..
echo
echo Creating chrnos_c
echo
cd chrnos_c
../../mkprodct.sh
cd ..
echo
echo Creating ckbref_c
echo
cd ckbref_c
../../mkprodct.sh
cd ..
echo
echo Creating commnt_c
echo
cd commnt_c
../../mkprodct.sh
cd ..
echo
echo Creating cook_c
echo
cd cook_c
../../mkprodct.sh
cd ..
echo
echo Creating dskbrief_c
echo
cd dskbrief_c
../../mkprodct.sh
cd ..
echo
echo Creating dskexp_c
echo
cd dskexp_c
../../mkprodct.sh
cd ..
echo
echo Creating frmdif_c
echo
cd frmdif_c
../../mkprodct.sh
cd ..
echo
echo Creating inspkt_c
echo
cd inspkt_c
../../mkprodct.sh
cd ..
echo
echo Creating mkdsk_c
echo
cd mkdsk_c
../../mkprodct.sh
cd ..
echo
echo Creating mkspk_c
echo
cd mkspk_c
../../mkprodct.sh
cd ..
echo
echo Creating msopck_c
echo
cd msopck_c
../../mkprodct.sh
cd ..
echo
echo Creating spacit_c
echo
cd spacit_c
../../mkprodct.sh
cd ..
echo
echo Creating spkdif_c
echo
cd spkdif_c
../../mkprodct.sh
cd ..
echo
echo Creating spkmrg_c
echo
cd spkmrg_c
../../mkprodct.sh
cd ..
echo
echo Creating tobin_c
echo
cd tobin_c
../../mkprodct.sh
cd ..
echo
echo Creating toxfr_c
echo
cd toxfr_c
../../mkprodct.sh
cd ..
echo
echo Creating versn_c
echo
cd versn_c
../../mkprodct.sh
cd ..
cd ..
echo Toolkit Build Complete

echo Making links
cd ./lib && ln -s cspice.a libcspice.a && ln -s csupport.a libcsupport.a && cd ..
echo Done making links

echo Making pkgconfig file
mkdir -p ./lib/pkgconfig && echo "prefix=\"`pwd`\"" > ./lib/pkgconfig/libcspice.pc && cat pkgconfig.stub >> ./lib/pkgconfig/libcspice.pc
echo Done making pkgconfig

echo All done
