### Generator

- **INPUT:**
	- 3-channel obrázek (256,256,3) - simulovaný výstup z obarvovací user appky *('ab' složky budou pro většinu pixelů prázdné/nulové, hodnoty budou mít pouze ty pixely, jež si user přeje obravit)*
- plus funkce generátoru schopna gerenovat šum, který bude třeba nějako konkatenovat se vstupem; na základě toho pak bude geneátor generovat

- **OUTPUT:**
	- 2-channel obrázek (256,256,2) - generátor vygeneruje barevné složky, které se pak následně zkonkatenují s 'L' složkou REAL obrázku *(do REAL grayscale obrázku se přidají vygenerované barvy, tím se obarví)*
	- Tj. output generátoru je 2-channel, ale úplný output této fáze je zase 3-channel (L\*a\*b) obrázek
	- *loss*

### Discriminator

- **INPUT:**
	- 3-channel REAL obrázek
	- 3-channel výstup generátorové fáze

- **OUTPUT:**
	- *prediction/loss (jak to má Emil)*