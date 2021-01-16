from CipherUtils import *

print("1A")
text = """
AOL ZPNUZ DLYL ZBIASL, HUK PA AVVR TL H DOPSL AV ZWVA AOLT, IBA NYHKBHSSF P ZAHYALK AV THRL AOLT VBA, HUK SPRL VUL VM AOVZL VSK MHZOPVULK 3K WPJABYLZ, AOHA ZWYPUNZ PUAV MVJBZ DOLU FVB JYVZZ FVBY LFLZ HUK JVBUA AV H OBUKYLK, AOL AYBAO JYFZAHSSPZLK HUK P YLHSPZLK AOHA P OHK ILLU ZLHYJOPUN MVY PA HSS HSVUN. PA DHZU’A AOHA P MVBUK ZVTLAOPUN WHYAPJBSHY. DOHA P UVAPJLK DHZ HJABHSSF HU HIZLUJL, H DOVSL JVSSLJAPVU VM HWWHYLUASF BUYLSHALK AOPUNZ AOHA ZOVBSK OHCL LEPZALK IBA KPKU’A. HUK QBZA HZ P OHK MPNBYLK AOHA VBA, ZVTLVUL, HUK IHJR AOLU P KPKU’A RUVD DOV, DYVAL AV ALSS TL HIVBA PA. AOLF VICPVBZSF OHK H ZLUZL VM AOL KYHTHAPJ, HUK HU LEJLSSLUA ZLUZL VM APTPUN. PM AOLF OHK ZLUA PA AV TL LCLU H MLD KHFZ ILMVYL P DVBSK OHCL HZZBTLK PA DHZ ZVTL RPUK VM JYHGF HKCLYAPZPUN ZABUA, IBA DOLU AOL WVZAJHYK HYYPCLK, PA DHZ PTTLKPHALSF VICPVBZ AV TL DOHA PA YLMLYYLK AV. PA JHYYPLK QBZA AOYLL DVYKZ, HUK PA KLZJYPILK WLYMLJASF AOL TPZZPUN WPLJLZ PU TF WBGGSL. PA QBZA ZHPK: AOL ZOHKVD HYJOPCL.
"""
Utils.get_stats(text)
print("Caesar Shift %s:%s" % Caesar.auto_decode(text))


"""
Caesar Shift 7:
THE SIGNS WERE SUBTLE, AND IT TOOK ME A WHILE TO SPOT THEM, BUT GRADUALLY I STARTED TO MAKE THEM OUT, AND LIKE ONE OF THOSE OLD FASHIONED 3D PICTURES, THAT SPRINGS INTO FOCUS WHEN YOU CROSS YOUR EYES AND COUNT TO A HUNDRED, THE TRUTH CRYSTALLISED AND I REALISED THAT I HAD BEEN SEARCHING FOR IT ALL ALONG. IT WASN’T THAT I FOUND SOMETHING PARTICULAR. WHAT I NOTICED WAS ACTUALLY AN ABSENCE, A WHOLE COLLECTION OF APPARENTLY UNRELATED THINGS THAT SHOULD HAVE EXISTED BUT DIDN’T. AND JUST AS I HAD FIGURED THAT OUT, SOMEONE, AND BACK THEN I DIDN’T KNOW WHO, WROTE TO TELL ME ABOUT IT. THEY OBVIOUSLY HAD A SENSE OF THE DRAMATIC, AND AN EXCELLENT SENSE OF TIMING. IF THEY HAD SENT IT TO ME EVEN A FEW DAYS BEFORE I WOULD HAVE ASSUMED IT WAS SOME KIND OF CRAZY ADVERTISING STUNT, BUT WHEN THE POSTCARD ARRIVED, IT WAS IMMEDIATELY OBVIOUS TO ME WHAT IT REFERRED TO. IT CARRIED JUST THREE WORDS, AND IT DESCRIBED PERFECTLY THE MISSING PIECES IN MY PUZZLE. IT JUST SAID: THE SHADOW ARCHIVE.
"""


print("1B")
text = """
JZFC XLUPDEJ,
TE SLD MPPY XJ OPPAPDE AWPLDFCP LYO L RCPLE ACTGTWPRP EZ DPCGP LD JZF ACTGLEP DPNCPELCJ ESPDP WLDE YTYP JPLCD, LYO ESZFRS TE HZFWO MP XJ LCOPYE HTDS EZ NZYETYFP EZ DPCGP JZF ESCZFRSZFE JZFC CPTRY, HP LCP, YZYP ZQ FD, TXXZCELW, LYO XJ ESZFRSED SLGP EFCYPO EZ XJ DFNNPDDZC.
JZF LCP ZQ NZFCDP PYETEWPO EZ OTDCPRLCO XJ LOGTNP, SZHPGPC T SLGP RTGPY NZYDTOPCLETZY EZ ESP NSLYRPD TY XJ CZWP ZGPC ESP WLDE DPGPCLW JPLCD. LD JZFC PXATCP SLD RCZHY TY XLRYTQTNPYNP, ZESPCD LNCZDD ZFC NZYETYPYE SLGP RCZHY QCLNETZFD, LYO LY TYNCPLDTYR AZCETZY ZQ XJ ETXP TD DAPYE XLYLRTYR ESP TXALNE ZQ ESPTC BFLCCPWD FAZY ZFC TDWLYO. T SLGP QPWE, LE ETXPD, WTVP DEPASPYDZY’D OC UPVJWW LD T SLGP XLYLRPO JZFC SZFDPSZWO LQQLTCD LYO ESP XZCP AFMWTN LDAPNED ZQ JZFC DELEP. LE ZESPCD T SLGP MPPY ACPDDPO EZ FYOPCELVP ESP CZWP ZQ XC SJOP, QZNFDDTYR XJ CLRP FYOPC RCPLE ACZGZNLETZY QCZX ESP AZHPCD ESLE ESCPLEPY EZ LDDLTW FD. HTES ESPDP CPQWPNETZYD T SLGP NZXP EZ ESP GTPH ESLE ESP ETXP XLJ SLGP NZXP EZ OTDDPNE ESP CZWP ZQ ACTGLEP DPNCPELCJ TYEZ TED EHZ GPCJ DPALCLEP QFYNETZYD.
ESP AFMWTN QLNP ZQ ESP CZJLW SZFDPSZWO XFDE ZQ NZFCDP NZYETYFP EZ MP ACPDPYEPO MJ DZXPZYP ZQ RCLNP LYO OTRYTEJ HSZ NLY NZXXLYO ESP NZYQTOPYNP ZQ ESP NZFCETPCD. HP SLGP OTDNFDDPO LE WPYRES HSZ XTRSE QTWW ESLE CZWP HSPY T PGPYEFLWWJ ALDD ZY, LYO T MPWTPGP ESLE HP SLGP LRCPPO EZ TYGTEP AZYDZYMJ EZ TYSPCTE ESLE XLYEWP. SP TD L RZZO XLY LYO HTWW DPCGP JZF HPWW. T HZFWO DFRRPDE ESLE QZC ESP DLVP ZQ NZYETYFTEJ SP NZYETYFPD EZ NLCCJ ESP ETEWP ZQ ACTGLEP DPNCPELCJ LYO HTWW MP SLAAJ EZ ACPALCP STX QZC ESTD CZWP. SZHPGPC ESPCP LCP DZXP LDAPNED ZQ XJ LNETGTETPD ESLE T DFDAPNE ESLE AZYDZYMJ HZFWO DECFRRWP EZ LNNZXAWTDS LYO QZC ESZDP T HZFWO TYGTEP JZF EZ NZYDTOPC L YPH AZDTETZY TY JZFC SZFDPSZWO, ESLE ZQ DPNCPE DPNCPELCJ.
TY ESPDP BFLCCPWDZXP ETXPD TE XLJ MP YPNPDDLCJ EZ NZXXTDDTZY LNETZYD ZC PYBFTCTPD ESLE DZXP XTRSE CPRLCO LD MPYPLES ESP OTRYTEJ ZQ ESP NCZHY. ESP DPNCPE DPNCPELCJ NLY, MJ NZYNPLWTYR ESPDP LNETGTETPD, ACPDPCGP ESP CPAFELETZY ZQ JZFC RZGPCYXPYE LD L CPWTLMWP LYO ECFDEHZCESJ ALCETNTALYE TY TYEPCYLETZYLW LQQLTCD, HSTWP LWDZ ACZGTOTYR JZF LYO JZFC XTYTDEPCD HTES ESP HPLAZYD EZ OPQPLE ZFC PYPXTPD. TQ HP DFNNPPO LD T SZAP HP HTWW, ESPY HLCD ZQ ESP QFEFCP XLJ MP HZY HTESZFE L DSZE MPTYR QTCPO.
TE TD XJ QPCGPYE SZAP ESLE JZF LRCPP HTES XJ LYLWJDTD LYO ESLE EZRPESPC HP NLY XZGP EZ PDELMWTDS ESP YPH ZQQTNP. T SLGP DPGPCLW YLXPD ESLE T HZFWO SFXMWJ DFRRPDE LD DECZYR NLYOTOLEPD QZC ESP YPH CZWP. LWW LCP RZZO XPY, HTES XTWTELCJ MLNVRCZFYOD LYO L CPAFELETZY QZC SZYZFC ESLE YZ-ZYP NZFWO BFPDETZY. T HTWW MP SLAAJ EZ OTDNFDD ESTD QFCESPC LE JZFC AWPLDFCP.
JZFC QLTESQFW DPCGLYE,
NSLCWPD RCPJ
"""
Utils.get_stats(text)
print("Caesar Shift %s:%s" % Caesar.auto_decode(text))


"""
Caesar Shift 11:
YOUR MAJESTY,
IT HAS BEEN MY DEEPEST PLEASURE AND A GREAT PRIVILEGE TO SERVE AS YOU PRIVATE SECRETARY THESE LAST NINE YEARS, AND THOUGH IT WOULD BE MY ARDENT WISH TO CONTINUE TO SERVE YOU THROUGHOUT YOUR REIGN, WE ARE, NONE OF US, IMMORTAL, AND MY THOUGHTS HAVE TURNED TO MY SUCCESSOR.
YOU ARE OF COURSE ENTITLED TO DISREGARD MY ADVICE, HOWEVER I HAVE GIVEN CONSIDERATION TO THE CHANGES IN MY ROLE OVER THE LAST SEVERAL YEARS. AS YOUR EMPIRE HAS GROWN IN MAGNIFICENCE, OTHERS ACROSS OUR CONTINENT HAVE GROWN FRACTIOUS, AND AN INCREASING PORTION OF MY TIME IS SPENT MANAGING THE IMPACT OF THEIR QUARRELS UPON OUR ISLAND. I HAVE FELT, AT TIMES, LIKE STEPHENSON’S DR JEKYLL AS I HAVE MANAGED YOUR HOUSEHOLD AFFAIRS AND THE MORE PUBLIC ASPECTS OF YOUR STATE. AT OTHERS I HAVE BEEN PRESSED TO UNDERTAKE THE ROLE OF MR HYDE, FOCUSSING MY RAGE UNDER GREAT PROVOCATION FROM THE POWERS THAT THREATEN TO ASSAIL US. WITH THESE REFLECTIONS I HAVE COME TO THE VIEW THAT THE TIME MAY HAVE COME TO DISSECT THE ROLE OF PRIVATE SECRETARY INTO ITS TWO VERY SEPARATE FUNCTIONS.
THE PUBLIC FACE OF THE ROYAL HOUSEHOLD MUST OF COURSE CONTINUE TO BE PRESENTED BY SOMEONE OF GRACE AND DIGNITY WHO CAN COMMAND THE CONFIDENCE OF THE COURTIERS. WE HAVE DISCUSSED AT LENGTH WHO MIGHT FILL THAT ROLE WHEN I EVENTUALLY PASS ON, AND I BELIEVE THAT WE HAVE AGREED TO INVITE PONSONBY TO INHERIT THAT MANTLE. HE IS A GOOD MAN AND WILL SERVE YOU WELL. I WOULD SUGGEST THAT FOR THE SAKE OF CONTINUITY HE CONTINUES TO CARRY THE TITLE OF PRIVATE SECRETARY AND WILL BE HAPPY TO PREPARE HIM FOR THIS ROLE. HOWEVER THERE ARE SOME ASPECTS OF MY ACTIVITIES THAT I SUSPECT THAT PONSONBY WOULD STRUGGLE TO ACCOMPLISH AND FOR THOSE I WOULD INVITE YOU TO CONSIDER A NEW POSITION IN YOUR HOUSEHOLD, THAT OF SECRET SECRETARY.
IN THESE QUARRELSOME TIMES IT MAY BE NECESSARY TO COMMISSION ACTIONS OR ENQUIRIES THAT SOME MIGHT REGARD AS BENEATH THE DIGNITY OF THE CROWN. THE SECRET SECRETARY CAN, BY CONCEALING THESE ACTIVITIES, PRESERVE THE REPUTATION OF YOUR GOVERNMENT AS A RELIABLE AND TRUSTWORTHY PARTICIPANT IN INTERNATIONAL AFFAIRS, WHILE ALSO PROVIDING YOU AND YOUR MINISTERS WITH THE WEAPONS TO DEFEAT OUR ENEMIES. IF WE SUCCEED AS I HOPE WE WILL, THEN WARS OF THE FUTURE MAY BE WON WITHOUT A SHOT BEING FIRED.
IT IS MY FERVENT HOPE THAT YOU AGREE WITH MY ANALYSIS AND THAT TOGETHER WE CAN MOVE TO ESTABLISH THE NEW OFFICE. I HAVE SEVERAL NAMES THAT I WOULD HUMBLY SUGGEST AS STRONG CANDIDATES FOR THE NEW ROLE. ALL ARE GOOD MEN, WITH MILITARY BACKGROUNDS AND A REPUTATION FOR HONOUR THAT NO-ONE COULD QUESTION. I WILL BE HAPPY TO DISCUSS THIS FURTHER AT YOUR PLEASURE.
YOUR FAITHFUL SERVANT,
CHARLES GREY
"""

