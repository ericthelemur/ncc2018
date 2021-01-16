from CipherUtils import *

print("2A")
text = """
SPML OHK ILLU YLSHAPCLSF KBSS ZPUJL YLABYUPUN MYVT TF DVYR DPAO OHYYF PU AOL TPKKSL LHZA. AOL IYPAPZO SPIYHYF DHZ KLSPNOALK AV NLA PAZ OHUKZ VU AOL YVTHU KPHYF HUK AOL JVSSLJAVY ZLLTLK AV OHCL MVYNVAALU HIVBA TL, THFIL ILJHBZL OHYYF’Z ALHT THKL PA AVV KHUNLYVBZ AV OHUN HYVBUK. P DHZ OHWWF AV ZLAASL IHJR PU AV TF DVYR HA AOL SPIYHYF HUK DHZ PUCVSCLK PU H WYVQLJA AV AYHJR KVDU HUK JHAHSVNBL TPZZPUN KVJBTLUAZ MYVT AOL SHAL UPULALLUAO JLUABYF. AOLF KLAHPSLK MVYLPNU WVSPJF, DOPJO ZVBUKZ KBSS, IBA DPAO CPJAVYPH AHRPUN HU HJAPCL PUALYLZA AOLYL DLYL H SVA VM SLAALYZ ILADLLU KVDUPUN ZAYLLA HUK AOL WHSHJL HUK P DHZ LUQVFPUN ZWFPUN VU MHTVBZ JOHYHJALYZ MYVT OPZAVYF. P YLHSSF MLSA SPRL P DHZ NLAAPUN ZVTL PUZPNOA PUAV OVD AOLF AOVBNOA HUK OVD AOL TVKLYU DVYSK JHTL PUAV ILPUN PU AOHA ABTBSABVBZ WLYPVK.

DOPSL P DHZ THPUSF AOLYL AV JOLJR AOL SLAALYZ MVY HBAOLUAPJPAF, P NVA YLHSSF PUCVSCLK PU AYFPUN AV BUKLYZAHUK OVD AOLF HSS MPAALK AVNLAOLY, HUK WHYA VM AOL QVI DHZ AV JYVZZ JOLJR ZAHALTLUAZ PU AOL SLAALYZ DPAO DOHA DL RUVD HJABHSSF OHWWLULK. AOLYL HYL H SVA VM WLVWSL DOV DPSS WHF H SVA VM TVULF AV VDU H SLAALY MYVT H YVFHS, ZV AOL HYJOPCL PZ WSHNBLK DPAO MVYNLYPLZ. ZVTL VM AOLT FVB JHU KLALJA IF HUHSFZPUN AOL WHWLY, VAOLYZ IF AOL DYPAPUN ZAFSL. ZVTL QBZA MHSS VCLY ILJHBZL AOL JVUALUA PZ VBA VM SPUL DPAO VAOLY KVJBTLUAZ, IBA HZ P ZABKPLK AOLT P ILNHU AV YLHSPZL AOHA H UBTILY VM AOLT OPUALK HA LCLUAZ AOHA P JVBSKU’A MPUK PU AOL OPZAVYPJHS YLJVYK. JLYAHPU UHTLZ HWWLHYLK HUK DLYL JSLHYSF PTWVYAHUA, HUK AOLU KPZHWWLHYLK JVTWSLALSF MYVT AYHJL. KPWSVTHAPJ PUJPKLUAZ DLYL TLUAPVULK AOHA ULCLY OHWWLULK HJJVYKPUN AV AOL OPZAVYF IVVRZ. VUL AOPUN FVB SLHYU PU AOPZ IBZPULZZ PZ AOHA AOL JPCPS ZLYCPJL ULCLY SLAZ HUF KLJPZPVU, OVDLCLY ZLJYLA, NV BUYLJVYKLK. VM JVBYZL AOHA TPNOA QBZA OHCL TLHUA AOVZL SLAALYZ HUK KVJBTLUAZ DLYL MHRL, IBA P WYPKL TFZLSM VU ILPUN HU LEJLSSLUA MVYNLY, HUK P DVBSK UVA OHCL ILLU HISL AV WYVKBJL AOLT. AOL WHWLY DHZ YPNOA, AOL PUR DHZ JOLTPJHSSF JVYYLJA HUK HNLK QBZA AOL YPNOA HTVBUA, HUK AOL ZAFSL VM DYPAPUN DHZ AVAHSSF JVUCPUJPUN. HUK P DHZ JVUCPUJLK. JVUCPUJLK AOHA ZVTLDOLYL AOLYL TBZA IL HU HYJOPCL VM NVCLYUTLUA KVJBTLUAZ MYVT AOL WLYPVK AOHA YLJVYKLK HSS VM AOLZL TPZZPUN ZAVYPLZ PU MBSS.

AOLU P YLJLPCLK AOL TLZZHNL HIVBA AOL ZOHKVD HYJOPCL. ZVTLVUL LSZL RULD HIVBA PA, HUK OHK DVYRLK VBA AOHA P DHZ OBUAPUN MVY PA AVV. AOL WVZAJHYK KPKU’A OLSW TBJO, IBA AOL LTHPSZ KPK. AOL MPYZA VUL OHK AOL ZBIQLJA SPUL QLRFSS HUK OFKL HUK DHZ LUJYFWALK BZPUN H ZPTWSL JHLZHY ZOPMA AV KPZJVBYHNL JHZBHS PUALYLZA. PA KPKU’A AHRL TL SVUN AV JYHJR PA, HUK AOL UHTLZ HUK KLAHPSZ PA JVUAHPULK THAJOLK AOL NYVDPUN SPZA VM TFZALYPVBZ YLMLYLUJLZ MYVT TF VDU YLZLHYJO. KVBNSHZ ISHJR DHZ JSLHYSF HU PTWVYAHUA MPNBYL, HUK P OHK H MLLSPUN AOHA OL OHK ZVTLAOPUN AV KV DPAO AOL HYJOPCL. AOHA MLLSPUN DHZ JVUMPYTLK IF AOL ZLJVUK LTHPS, ISHJR OLHYA, AOHA P YLJLPCLK SHALY AOHA DLLR. HNHPU PA DHZ LUJYFWALK IBA AOPZ APTL BZPUN HU HMMPUL ZOPMA JPWOLY. PA DHZ JSLHYSF MYVT AOL ZHTL PUKPCPKBHS - HA AOL CLYF SLHZA DOVLCLY DHZ ZLUKPUN TL AOL LTHPSZ OHK H OHIPA VM TPZZPUN AOL SLAALY Y MYVT AOL DVYK "FVBY".

ZVTLVUL DHZ WSHFPUN NHTLZ DPAO TL, HUK P DHZ TVYL AOHU OHWWF AV QVPU PU.
"""
Utils.get_stats(text)
print("%s%s" % Caesar.auto_decode(text))

"""
Caesar Shift 7
LIFE HAD BEEN RELATIVELY DULL SINCE RETURNING FROM MY WORK WITH HARRY IN THE MIDDLE EAST. THE BRITISH LIBRARY WAS DELIGHTED TO GET ITS HANDS ON THE ROMAN DIARY AND THE COLLECTOR SEEMED TO HAVE FORGOTTEN ABOUT ME, MAYBE BECAUSE HARRY’S TEAM MADE IT TOO DANGEROUS TO HANG AROUND. I WAS HAPPY TO SETTLE BACK IN TO MY WORK AT THE LIBRARY AND WAS INVOLVED IN A PROJECT TO TRACK DOWN AND CATALOGUE MISSING DOCUMENTS FROM THE LATE NINETEENTH CENTURY. THEY DETAILED FOREIGN POLICY, WHICH SOUNDS DULL, BUT WITH VICTORIA TAKING AN ACTIVE INTEREST THERE WERE A LOT OF LETTERS BETWEEN DOWNING STREET AND THE PALACE AND I WAS ENJOYING SPYING ON FAMOUS CHARACTERS FROM HISTORY. I REALLY FELT LIKE I WAS GETTING SOME INSIGHT INTO HOW THEY THOUGHT AND HOW THE MODERN WORLD CAME INTO BEING IN THAT TUMULTUOUS PERIOD.

WHILE I WAS MAINLY THERE TO CHECK THE LETTERS FOR AUTHENTICITY, I GOT REALLY INVOLVED IN TRYING TO UNDERSTAND HOW THEY ALL FITTED TOGETHER, AND PART OF THE JOB WAS TO CROSS CHECK STATEMENTS IN THE LETTERS WITH WHAT WE KNOW ACTUALLY HAPPENED. THERE ARE A LOT OF PEOPLE WHO WILL PAY A LOT OF MONEY TO OWN A LETTER FROM A ROYAL, SO THE ARCHIVE IS PLAGUED WITH FORGERIES. SOME OF THEM YOU CAN DETECT BY ANALYSING THE PAPER, OTHERS BY THE WRITING STYLE. SOME JUST FALL OVER BECAUSE THE CONTENT IS OUT OF LINE WITH OTHER DOCUMENTS, BUT AS I STUDIED THEM I BEGAN TO REALISE THAT A NUMBER OF THEM HINTED AT EVENTS THAT I COULDN’T FIND IN THE HISTORICAL RECORD. CERTAIN NAMES APPEARED AND WERE CLEARLY IMPORTANT, AND THEN DISAPPEARED COMPLETELY FROM TRACE. DIPLOMATIC INCIDENTS WERE MENTIONED THAT NEVER HAPPENED ACCORDING TO THE HISTORY BOOKS. ONE THING YOU LEARN IN THIS BUSINESS IS THAT THE CIVIL SERVICE NEVER LETS ANY DECISION, HOWEVER SECRET, GO UNRECORDED. OF COURSE THAT MIGHT JUST HAVE MEANT THOSE LETTERS AND DOCUMENTS WERE FAKE, BUT I PRIDE MYSELF ON BEING AN EXCELLENT FORGER, AND I WOULD NOT HAVE BEEN ABLE TO PRODUCE THEM. THE PAPER WAS RIGHT, THE INK WAS CHEMICALLY CORRECT AND AGED JUST THE RIGHT AMOUNT, AND THE STYLE OF WRITING WAS TOTALLY CONVINCING. AND I WAS CONVINCED. CONVINCED THAT SOMEWHERE THERE MUST BE AN ARCHIVE OF GOVERNMENT DOCUMENTS FROM THE PERIOD THAT RECORDED ALL OF THESE MISSING STORIES IN FULL.

THEN I RECEIVED THE MESSAGE ABOUT THE SHADOW ARCHIVE. SOMEONE ELSE KNEW ABOUT IT, AND HAD WORKED OUT THAT I WAS HUNTING FOR IT TOO. THE POSTCARD DIDN’T HELP MUCH, BUT THE EMAILS DID. THE FIRST ONE HAD THE SUBJECT LINE JEKYLL AND HYDE AND WAS ENCRYPTED USING A SIMPLE CAESAR SHIFT TO DISCOURAGE CASUAL INTEREST. IT DIDN’T TAKE ME LONG TO CRACK IT, AND THE NAMES AND DETAILS IT CONTAINED MATCHED THE GROWING LIST OF MYSTERIOUS REFERENCES FROM MY OWN RESEARCH. DOUGLAS BLACK WAS CLEARLY AN IMPORTANT FIGURE, AND I HAD A FEELING THAT HE HAD SOMETHING TO DO WITH THE ARCHIVE. THAT FEELING WAS CONFIRMED BY THE SECOND EMAIL, BLACK HEART, THAT I RECEIVED LATER THAT WEEK. AGAIN IT WAS ENCRYPTED BUT THIS TIME USING AN AFFINE SHIFT CIPHER. IT WAS CLEARLY FROM THE SAME INDIVIDUAL - AT THE VERY LEAST WHOEVER WAS SENDING ME THE EMAILS HAD A HABIT OF MISSING THE LETTER R FROM THE WORD "YOUR".
"""


print("2B")
text = """
GYN OFCNDAG,

YZ YG EYZF NAMNAZ ZFCZ Y TYPH WQGADT YP HYGCMNAAWAPZ EYZF QIS RSHMAWAPZ. YZ YG ODACN ZI WA ZFCZ QIS CNA NYMFZ YP IPA NAMCNH, YZ YG ZYWA ZI AGZCVDYGF ZFA ITTYOA IT GAONAZ GAONAZCNQ, CPH ZI ZCKA ZFA GAONAZ ECN ZI ISN APAWYAG. IP IPA GYMPYTYOCPZ BIYPZ FIEALAN, Y HI PIZ CMNAA. ZFYG NIDA YG PIZ GSYZAH ZI MIIH WAP EYZF C NABSZCZYIP TIN FIPISN. QISN GSMMAGZYIPG EISDH VA OCBYZCD YT Y EANA DIIKYPM ZI CBBIYPZ C OFYAT IT GZCTT IN C PAE TINAYMP GAONAZCNQ, FIEALAN ZFA ZCGKG ZFCZ EA VIZF KPIE CNA PAOAGGCNQ YT EA CNA ZI BNIZAOZ CPH AXBCPH ZFA AWBYNA EYDD NAUSYNA C WCP IT CDZIMAZFAN HYTTANAPZ OFCNCOZAN. C NAH VDIIHAH WCP EYZF C VDCOK FACNZ.

ZFANA YG IPA WCP EA VIZF KPIE EFI YG APZYNADQ GSYZAH ZI ZFA DAGG OIPMAPYCD CGBAOZG IT WIHANP GZCZAONCTZ, CPH Y CW GSNBNYGAH ZFCZ QIS HYH PIZ CHH FYG PCWA ZI ZFA DYGZ - QIS CWANYOCP OISGYP HISMDCG VDCOK. VDCOK YG C WCP IT GYPMSDCN ZCDAPZG CPH Y EISDH GSMMAGZ ZFCZ QIS COZ EYZF SZWIGZ GBAAH ZI VNYPM FYW ZI DIPHIP. Y VADYALA ZFCZ FA YG OSZ TNIW ZFA GCWA ODIZF CG QIS, CPH Y CW OIPTYHAPZ ZFCZ QIS EYDD VA CVDA ZI BANGSCHA FYW ZI ZCKA SB ZFA BIGZ IT GAONAZ GAONAZCNQ. Y CW NCZFAN DIIKYPM TINECNH ZI WQ TYNGZ WAAZYPM EYZF WN. VDCOK CPH Y ZNSGZ QIS EYDD PIZ HYGCBBIYPZ WA YP ZFYG, VSZ YT QIS TYPH ZFCZ FA YG PIZ CWAPCVDA ZI NACGIP ZFAP Y EYDD TYPH CPIZFAN ECQ ZI BANGSCHA FYW. C WCP DYKA VDCOK CDECQG FCG C GKADAZIP IN ZEI YP FYG ODIGAZ!

L.
"""
Utils.get_stats(text)
print("%s%s" % Affine.auto_decode(text))

"""
Affine Shift a = 19, b = 2
SIR CHARLES,

IT IS WITH REGRET THAT I FIND MYSELF IN DISAGREEMENT WITH YOU JUDGEMENT. IT IS CLEAR TO ME THAT YOU ARE RIGHT IN ONE REGARD, IT IS TIME TO ESTABLISH THE OFFICE OF SECRET SECRETARY, AND TO TAKE THE SECRET WAR TO OUR ENEMIES. ON ONE SIGNIFICANT POINT HOWEVER, I DO NOT AGREE. THIS ROLE IS NOT SUITED TO GOOD MEN WITH A REPUTATION FOR HONOUR. YOUR SUGGESTIONS WOULD BE CAPITAL IF I WERE LOOKING TO APPOINT A CHIEF OF STAFF OR A NEW FOREIGN SECRETARY, HOWEVER THE TASKS THAT WE BOTH KNOW ARE NECESSARY IF WE ARE TO PROTECT AND EXPAND THE EMPIRE WILL REQUIRE A MAN OF ALTOGETHER DIFFERENT CHARACTER. A RED BLOODED MAN WITH A BLACK HEART.

THERE IS ONE MAN WE BOTH KNOW WHO IS ENTIRELY SUITED TO THE LESS CONGENIAL ASPECTS OF MODERN STATECRAFT, AND I AM SURPRISED THAT YOU DID NOT ADD HIS NAME TO THE LIST - YOU AMERICAN COUSIN DOUGLAS BLACK. BLACK IS A MAN OF SINGULAR TALENTS AND I WOULD SUGGEST THAT YOU ACT WITH UTMOST SPEED TO BRING HIM TO LONDON. I BELIEVE THAT HE IS CUT FROM THE SAME CLOTH AS YOU, AND I AM CONFIDENT THAT YOU WILL BE ABLE TO PERSUADE HIM TO TAKE UP THE POST OF SECRET SECRETARY. I AM RATHER LOOKING FORWARD TO MY FIRST MEETING WITH MR. BLACK AND I TRUST YOU WILL NOT DISAPPOINT ME IN THIS, BUT IF YOU FIND THAT HE IS NOT AMENABLE TO REASON THEN I WILL FIND ANOTHER WAY TO PERSUADE HIM. A MAN LIKE BLACK ALWAYS HAS A SKELETON OR TWO IN HIS CLOSET!

V.
"""