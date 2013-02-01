#!/bin/bash

#################################################################
export PATH=$PATH:../Posda/bin
export PATH=$PATH:../Posda/bin/ae
export PATH=$PATH:../Posda/bin/contrib

PRG1=../Posda/bin/AnonymizeSearch.pl
PRG2=../Posda/bin/Anonymize.pl
#################################################COMMENTARY


if [ "$#" = "0" ]
then clear
echo "############################################################" 
echo "This programm could anonymize dicom and vxp files"
echo "It was developped by Jeremy ASSAYH 09/08 in SCI Institue and U OF U Radiation Oncology"
echo "University of Utah, Salt Lake city"
echo "Version Beta 1.0"
echo "To use it : "
echo "./anonymizeDicomRpm.sh <input directory> <output directory> <Anonym ID> <Anonym Name>"
echo "############################################################" 
exit
fi


echo "Welcome to the Dicom Anonynizer software"
echo "Version Beta 1.0"
###########################################################CREATION OF THE MAP
#OPEN THE DIRECTORY WHERE THE SCANNER IS
#WE CHECK IF WE PUT / OR NOT
longueur=${#1}
end=${1:longueur-1:1}

if [ "$end" = "/" ]
    then SRC=${1:0:longueur-1}
else
    SRC=$1
fi

longueur=${#2}
end=${2:longueur-1:1}
if [ "$end" = "/" ]
    then OUTPUT=${2:0:longueur-1}
else
    OUTPUT=$2
fi

#CREATION OF THE MAP
$PRG1 $SRC > $OUTPUT/map.txt
echo "Map created : OK"

#IF NO DICOM IMAGES
NBD=`ls -s $OUTPUT/map.txt | awk -F' ' '{print $1}'`
if [ "$NBD" = "0" ]
    then echo "No Dicom Images in the directory" 
    exit
fi

###########################################################WE HAVE TO MODIFY THE MAP.TXT
#KEEP INFORMATION
cat $OUTPUT/map.txt | \
grep -v "|Patient's Sex|" | \
grep -v "|Study Description|" | \
grep -v "|Series Description|" | \
grep -v "|Series Description|" | \
grep -v "|Institution Name|" | \
grep -v "|Referring Physician's Name|" | \
grep -v "|Station Name|" | \
grep -v "|Station Name|" | \
grep -v "|Operator's Name|" | \
grep -v "|Protocol Name|" | \
grep -v "|Performed Procedure Step ID|" | \
grep -v "|Structure Set Label|" | \
grep -v "|Structure Set Name|" | \
grep -v "|Name of Physician(s) Reading Study|" | \
grep -v "|Study ID|" | \
grep -v "|Patient's Age|" > $OUTPUT/new_map.txt


#THE ID
line_id=$(grep "Patient's ID" $OUTPUT/new_map.txt)
longueur1=${#line_id}
A=${line_id:0:longueur1-1}
B=$3
C='"'
new_line_id=$A$B$C
sed -i "s/^\(.*Patient's ID.*=>\).*$/\1 \"$B\"/" $OUTPUT/new_map.txt

old_id1=`grep "Patient's ID" $OUTPUT/new_map.txt | awk -F'"' '{print $2}'| awk -F'/' '{print $1}'`
old_id2=`grep "Patient's ID" $OUTPUT/new_map.txt | awk -F'"' '{print $2}'| awk -F'/' '{print $2}'`
old_id=$old_id1$old_id2


#THE NAME
line_name=$(grep "Patient's Name" $OUTPUT/new_map.txt)
longueur2=${#line_name}
A=${line_name:0:longueur2-1}
B=$4
C='"'
new_line_name=$A$B$C
sed -i "s/$line_name/$new_line_name/" $OUTPUT/new_map.txt


#THE DATE
NB_line_map=$(grep -c "date_map" $OUTPUT/new_map.txt)
line=`grep -n "date_map" $OUTPUT/new_map.txt | awk '{print FNR":" $1}' | grep -E ^1 | awk -F':' '{print $2}'`

debut=$((line+1))
fin=$((line+NB_line_map-1))

for ((i=$debut;i<=$fin;i=i+1));
do
  line_date=`cat $OUTPUT/new_map.txt | awk '{print FNR":"$0}' | grep -E ^$i | awk -F':' '{print $2}'`
  old_date=`cat $OUTPUT/new_map.txt | awk '{print FNR":"$0}' | grep -E ^$i | awk -F'"' '{print $2}'`
  A="2001"
  B=${old_date:4:4}
  new_date=$A$B
  longueur3=${#line_date}
  D=${line_date:0:longueur3-1}
  E=$new_date
  F='"'
  new_line_date=$D$E$F
  sed  -i "s/$line_date/$new_line_date/" $OUTPUT/new_map.txt
done

#BIRTHDAY DATE
line_date=`cat $OUTPUT/new_map.txt | awk '{print FNR":"$0}' | grep -E ^$line | awk -F':' '{print $2}'`
new_date="09112001"
D=${line_date:0:longueur3-1}
E=$new_date
F='"'
new_line_date=$D$E$F
sed  -i "s/$line_date/$new_line_date/" $OUTPUT/new_map.txt

echo "The map modified : OK"

############################################################ANONYMIZE
mkdir $OUTPUT/$3
$PRG2 $OUTPUT/new_map.txt $SRC $OUTPUT/$3 >> /dev/null
echo "Anonymize : OK"

##############################################################VXP FILES
NB=`ls $SRC/*.vxp | wc -l`

for ((j=1; j <= $NB; j+=1))
do
  
  #NAME OF THE VXP FILES
  name=`ls $SRC/*.vxp | awk '{print FNR":"$0}' | grep -E ^$j`
  longueur=${#SRC}
  name=${name:2}
  name=${name:longueur+1}
    
  #OPEN VXP FILES
  vxp_files=$SRC/$name
  tr -d '\015' < $vxp_files > $OUTPUT/${name}.tmp
  vxp_files=$OUTPUT/$name
  
  #DATE
  longueur2=5
  line_date=$(grep "Date" ${vxp_files}.tmp)
  date=${line_date:$longueur2}
  tmp1="2001"
  anonym_date=${date:0:6}$tmp1
  sed -i "s/Date=.*/Date=$anonym_date/" ${vxp_files}.tmp
  
  #ID
  id=`cat ${vxp_files}.tmp | grep "Patient_ID" | awk -F= '{print $2}'`

  if [ "$id" != "$old_id" ]
      then echo "wrong files beacause the id is different"
      rm ${vxp_files}.tmp
  else
      anonym_id=$3
      sed -i "s/Patient_ID=.*/Patient_ID=$anonym_id/" ${vxp_files}.tmp
  
      #RENAME THE FILE
      extension=".vxp"
      entre="_"
      name_files=$anonym_id$entre$anonym_date$extension
      mv ${vxp_files}.tmp $OUTPUT/$name_files
      echo $name_files
  fi
  
done
echo "Vxp_files : OK"
rm $OUTPUT/new_map.txt
rm $OUTPUT/map.txt
