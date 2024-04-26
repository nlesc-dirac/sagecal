#!/bin/bash


# find .pgm files (if main.x is given)
if [ -f main.x ]; then
if compgen -G "*.pgm" > /dev/null; then
 for MAIN in *.pgm; do
  echo $MAIN;
  STEM=`echo $MAIN|sed 's/.pgm//g'`
  TARGET=$STEM.px
  cp $MAIN $STEM"_main.c"
  cp main.x $TARGET
 done
fi
fi

CC=gcc
if ! [ -f main.x ]; then
 CFLAGS="-c -ansi -m64 -O2 -DNON_UNIX_STDIO"
else
 CFLAGS="-c -ansi -m64 -O2 -fPIC -DNON_UNIX_STDIO"
fi
LDFLAGS="-lm -m64"
# set compiler to gcc
TKCOMPILER=$CC
TKCOMPILEOPTIONS=$CFLAGS
TKLINKOPTIONS=$LDFLAGS


# library name is current dir
DIR=$(basename `pwd`)
LIBRARY="../../lib/"$DIR


# check any *.c files need compiling
if compgen -G "*.c" > /dev/null; then
 for SRCFILE in *.c; do
   $TKCOMPILER $TKCOMPILEOPTIONS $SRCFILE
 done
fi

# if program files exit, need to create a library (override LIBRARY above)
if compgen -G "*.pgm" > /dev/null; then
  LIBRARY="locallib"
fi

if compgen -G "*.o" > /dev/null; then
  echo "linking library $LIBRARY"
  ar crv $LIBRARY.a *.o
  ranlib $LIBRARY.a
  rm *.o
fi


# if there are programs, compile them
if compgen -G "*.pgm" > /dev/null; then
  # if main.x is not present, go through *.pgm, not *.px files
  if ! [ -f main.x ]; then
    for MAIN in *.pgm; do
      STEM=`echo $MAIN|sed 's/.pgm//g'`
      TARGET=$STEM.c
      MAINOBJ=$STEM.o
      EXECUT="../../exe/"$STEM
      cp $MAIN $TARGET
      echo "compiling and linking $MAIN"
      if [ -f locallib.a ]; then
        $TKCOMPILER $TKCOMPILEOPTIONS $TARGET
        $TKCOMPILER -o $EXECUT $MAINOBJ \
          locallib.a \
          ../../lib/csupport.a \
          ../../lib/cspice.a \
          $TKLINKOPTIONS

        rm $TARGET
        rm $MAINOBJ
        rm locallib.a
      else
        $TKCOMPILER $TKCOMPILEOPTIONS $TARGET
        $TKCOMPILER -o $EXECUT $MAINOBJ \
          ../../lib/csupport.a \
          ../../lib/cspice.a \
          $TKLINKOPTIONS

        rm $TARGET
        rm $MAINOBJ
      fi
    done
  else
    for MAIN in *.px; do
      STEM=`echo $MAIN|sed 's/.px//g'`
      TARGET=$STEM.c
      MAINOBJ=$STEM.o
      EXECUT="../../exe/"$STEM
      cp $MAIN $TARGET
      echo "compiling and linking $MAIN"
      if [ -f locallib.a ]; then
        $TKCOMPILER $TKCOMPILEOPTIONS $TARGET
        $TKCOMPILER -o $EXECUT $MAINOBJ \
          locallib.a \
          ../../lib/csupport.a \
          ../../lib/cspice.a \
          $TKLINKOPTIONS

        rm $TARGET
        rm $MAINOBJ
        rm locallib.a
      else
        $TKCOMPILER $TKCOMPILEOPTIONS $TARGET
        $TKCOMPILER -o $EXECUT $MAINOBJ \
          ../../lib/csupport.a \
          ../../lib/cspice.a \
          $TKLINKOPTIONS

        rm $TARGET
        rm $MAINOBJ
      fi
    done
  fi
fi



# cleanup
if compgen -G "*.o" > /dev/null; then
 rm *.o
fi
if compgen -G "*.px" > /dev/null; then
 rm *.px
fi
if compgen -G "*_main.c" > /dev/null; then
 rm *_main.c
fi
