fewshot_data = {'DE': """### Source Text:
Identify the main topic of the paragraph. <x>SEP</x> The discovery of the structure of DNA in 1953 is one of the most significant scientific achievements in history. The structure of DNA, a double helix, was discovered by James Watson and Francis Crick, with the help of Rosalind Franklin's X-ray diffraction images. This discovery has had a profound impact on the field of genetics and has led to many important scientific breakthroughs. <x>SEP</x> The main topic of the paragraph is the discovery of the structure of DNA.

### Translation:
Bestimmen Sie das Hauptthema des Absatzes. <x>SEP</x> Die Entdeckung der Struktur der DNA im Jahr 1953 gehört zu den bedeutendsten wissenschaftlichen Errungenschaften der Geschichte. Die Struktur der DNA, eine Doppelhelix, wurde von James Watson und Francis Crick mithilfe von Rosalind Franklins Röntgenbeugungsbildern entdeckt. Diese Entdeckung hatte tiefgreifende Auswirkungen auf das Gebiet der Genetik und führte zu vielen wichtigen wissenschaftlichen Durchbrüchen. <x>SEP</x> Das Hauptthema des Absatzes ist die Entdeckung der Struktur der DNA.##EOT##

### Source Text:
Calculate the area of a rectangle with a length of 5 inches and a width of 3 inches, and with a length of 7 inches and a width of 2 inches, and then find the average area. <x>SEP</x> length_1 = 5, width_1 = 3, length_2 = 7, width_2 = 2 <x>SEP</x> To find the area of each rectangle, we multiply the length by the width.

Rectangle 1: Area = length_1 × width_1 = 5 × 3 = 15 square inches
Rectangle 2: Area = length_2 × width_2 = 7 × 2 = 14 square inches

To find the average area, we add the areas of both rectangles and divide by 2:

Average Area = (15 + 14) / 2 = 29 / 2 = 14.5 square inches

So, the average area of the two rectangles is 14.5 square inches.

### Translation:
Berechne die Fläche eines Rechtecks mit einer Länge von 5 Zoll und einer Breite von 3 Zoll sowie eines Rechtecks mit einer Länge von 7 Zoll und einer Breite von 2 Zoll und ermittle dann die durchschnittliche Fläche. <x>SEP</x> Länge_1 = 5, Breite_1 = 3, Länge_2 = 7, Breite_2 = 2 <x>SEP</x> Um die Fläche jedes Rechtecks zu berechnen, multiplizieren wir die Länge mit der Breite.

Rechteck 1: Fläche = Länge_1 × Breite_1 = 5 × 3 = 15 Quadratzoll
Rechteck 2: Fläche = Länge_2 × Breite_2 = 7 × 2 = 14 Quadratzoll

Um die durchschnittliche Fläche zu berechnen, addieren wir die Flächen beider Rechtecke und teilen durch 2:

Durchschnittliche Fläche = (15 + 14) / 2 = 29 / 2 = 14,5 Quadratzoll

Die durchschnittliche Fläche der beiden Rechtecke beträgt also 14,5 Quadratzoll.##EOT##""",

'DE_judge': """# Example 1:
## Source text:
Write a short paragraph about the history of the city of Rome.
Rome, the \"Eternal City,\" is steeped in history. Founded, according to legend, by Romulus and Remus in 753 BC, Rome has been a major human settlement for over two and a half millennia. It began as a small Italian settlement and expanded to become the center of a vast empire that dominated the Mediterranean region for centuries. The Roman Empire was known for its military prowess, political hegemony, remarkable engineering, and cultural contributions. After the fall of the Western Roman Empire in the 5th century AD, Rome endured a period of fragmentation and decline. However, it regained importance during the Middle Ages and the Renaissance as the heart of Catholicism and a center of arts and learning. Today, Rome serves as the capital city of modern Italy.

## Translation (a):
Schreiben Sie einen kurzen Absatz über die Geschichte der Stadt Rom.
Rom, die \"Ewige Stadt\", ist reich an Geschichte. Laut der Legende wurde sie im Jahr 753 v. Chr. von Romulus und Remus gegründet und ist seit über zweieinhalb Jahrtausenden ein bedeutendes menschliches Siedlungsgebiet. Sie begann als kleine italienische Siedlung und dehnte sich aus, um das Zentrum eines riesigen Reiches zu werden, das jahrhundertelang das Mittelmeergebiet dominierte. Das Römische Reich war bekannt für seine militärische Stärke, politische Vorherrschaft, bemerkenswerte Ingenieurskunst und kulturelle Beiträge. Nach dem Untergang des Weströmischen Reiches im 5. Jahrhundert n. Chr. erlebte Rom eine Zeit der Zersplitterung und des Niedergangs. Es gewann jedoch während des Mittelalters und der Renaissance wieder an Bedeutung als Zentrum des Katholizismus und als Mittelpunkt von Kunst und Wissenschaft. Heute dient Rom als Hauptstadt des modernen Italiens.

## Translation (b):
Schreibe einen kurzen Absatz über die Geschichte der Stadt Rom.
Rom, die \"ewige Stadt\", ist in der Geschichte verwurzelt. Gemäß der Legende wurde sie 753 v. Chr. von Romulus und Remus gegründet und war seit mehr als zwei und einem halben Jahrtausend ein wichtiger menschlicher Siedlungsort. Sie begann als kleiner italienischer Ort und erweiterte sich, um das Zentrum eines riesigen Imperiums zu werden, das Jahrzehnte lang die Mittelmeerregion dominierte. Das Römische Reich war für seine militärische Stärke, politische Hegemonie, bemerkenswerte Ingenieurkunst und kulturelle Beiträge bekannt. Nach dem Untergang des Weströmischen Reiches im 5. Jahrhundert n. Chr. erlebte Rom eine Zeit der Fragmentierung und des Niedergangs. Allerdings erlangte es während des Mittelalters und der Renaissance als Herzstück des Katholizismus und als Zentrum der Kunst und des Lernens wieder Bedeutung. Heute ist Rom die Hauptstadt des modernen Italiens.

## Which is best, Translation (a) or Translation (b)?
Translation (a) ##EOS##

Here the answer is Translation (a) because it is more accurate and stylistically appropriate. It uses correct terminology, such as "Ewige Stadt" (capitalized to reflect its status as a title), which aligns well with the source's "Eternal City." The phrase "zwei und einem halben Jahrtausend" in (b) is slightly awkward in German, whereas (a) more naturally says "über zweieinhalb Jahrtausenden." Also, (a) uses "jahrhundertelang" for "for centuries," which is a better fit than "Jahrzehnte lang" ("decades long") in (b), preserving the sense of Rome's extensive historical dominance. Overall, (a) is smoother and more precise.

# Example 2:
## Source text:
Write a code to calculate the average of a list of numbers
Here is a Python code snippet to calculate the average of a list of numbers.\n\n```python\ndef calculate_average(numbers):\n    return sum(numbers) / len(numbers)\n\n# test the function with an example list\nnumbers = [4, 5, 6, 7, 8, 9]\nprint(\"The average is\", calculate_average(numbers))\n```\n\nThis code defines a function `calculate_average()` that calculates the average of a list of numbers. It first sums up all the numbers in the list using Python's built-in `sum()` function, then divides the sum by the number of elements in the list, which is obtained using Python's `len()` function. The result, which is the average, is then returned.\n\nKeep in mind that Python 3's division operation `/` results in a float number, while Python 2's division operation `/` results in an integer if both operands are integers. If you're working with Python 2 and you want a float division, you can convert one operand to float manually, or use `from __future__ import division` at the beginning of your code for float division to become the default behavior.\n\nThis code also includes a testing portion where an example list of numbers is given to the function, and the resulting average is printed out. You can replace this with your own list or use the function in different places in your program as needed. Make sure to pass a non-empty list of numbers to this function. An empty list will result in a division by zero error.\n\n### Output:\n\n```\nThe average is 6.5\n```\n\nThis result is the average of the numbers in the given list [4, 5, 6, 7, 8, 9], calculated by the Python code. If you provide a different list of numbers, the function will return a different average.

## Translation (b):
Schreibe ein Codebeispiel, um den Durchschnitt einer Liste von Zahlen zu berechnen.
Hier ist ein Python-Code-Snippet, um den Durchschnitt einer Liste von Zahlen zu berechnen.\n\n```python\ndef calculate_average(numbers):\n    return sum(numbers) / len(numbers)\n\n# teste die Funktion mit einem Beispiel-Liste\nnumbers = [4, 5, 6, 7, 8, 9]\nprint(\"Der Durchschnitt beträgt\", calculate_average(numbers))\n```\n\nDieser Code definiert eine Funktion `calculate_average()`, die den Durchschnitt einer Liste von Zahlen berechnet. Zunächst summiert er alle Zahlen in der Liste mithilfe von Pythons integrierter `sum()`-Funktion, dann teilt er die Summe durch die Anzahl der Elemente in der Liste, die mit Pythons `len()`-Funktion ermittelt wird. Das Ergebnis, also der Durchschnitt, wird dann zurückgegeben.\n\nAchtung: Die Division in Python 3 mit `/` ergibt ein float-Zahl, während die Division in Python 2 mit `/` ein Integer ergibt, wenn beide Operanden Integers sind. Wenn Sie mit Python 2 arbeiten und ein float-Division erhalten möchten, können Sie ein Operand manuell in float umwandeln oder verwenden Sie `from __future__ import division` am Anfang Ihres Codes, um float-Division zum Standardverhalten zu machen.\n\nDieser Code enthält auch ein Test-Portion, in dem eine Beispiel-Liste von Zahlen der Funktion zugeordnet wird, und der ergebende Durchschnitt ausgegeben wird. Sie können Ihre eigene Liste oder die Funktion in anderen Teilen Ihres Programms verwenden, wie Sie es wünschen. Stellen Sie sicher, dass Sie eine nicht-leere Liste von Zahlen dieser Funktion zuordnen. Eine leere Liste wird zu einer Division durch Null führen.

## Translation (a):
Schreiben Sie einen Code, um den Durchschnitt einer Liste von Zahlen zu berechnen. 
Hier ist ein Python-Code-Schnipsel, um den Durchschnitt einer Liste von Zahlen zu berechnen.\n\n```python\ndef calculate_average(numbers):\n    return sum(numbers) / len(numbers)\n\n# test the function with an example list\nnumbers = [4, 5, 6, 7, 8, 9]\nprint(\"The average is\", calculate_average(numbers))\n```\n\nDieser Code definiert eine Funktion `calculate_average()`, die den Durchschnitt einer Liste von Zahlen berechnet. Zuerst werden alle Zahlen in der Liste mit Hilfe der Python-integrierten Funktion `sum()` aufsummiert, dann wird die Summe durch die Anzahl der Elemente in der Liste geteilt, die mit Hilfe der Python-Funktion `len()` erhalten wird. Das Ergebnis, das der Durchschnitt ist, wird dann zurückgegeben.\n\nBeachten Sie, dass die Python 3-Divisionsoperation `/` ein Gleitkommazahl-Ergebnis liefert, während die Python 2-Divisionsoperation `/` ein ganzzahliges Ergebnis liefert, wenn beide Operanden ganze Zahlen sind. Wenn Sie mit Python 2 arbeiten und eine Gleitkomma-Division wünschen, können Sie einen Operanden manuell in eine Gleitkommazahl umwandeln oder `from __future__ import division` am Anfang Ihres Codes verwenden, damit die Gleitkomma-Division zum Standardverhalten wird.\n\nDieser Code enthält auch einen Testabschnitt, in dem eine Beispiel-Liste von Zahlen an die Funktion übergeben wird und das resultierende Mittelwert ausgegeben wird. Sie können dies durch Ihre eigene Liste ersetzen oder die Funktion an verschiedenen Stellen in Ihrem Programm verwenden, wie es erforderlich ist. Stellen Sie sicher, dass Sie eine nicht leere Liste von Zahlen an diese Funktion übergeben. Eine leere Liste führt zu einem Division durch Null-Fehler.\n\n### Output:\n\n```\nThe average is 6.5\n```\n\nDieses Ergebnis ist der Durchschnitt der Zahlen in der gegebenen Liste [4, 5, 6, 7, 8, 9], berechnet durch den Python-Code. Wenn Sie eine andere Liste von Zahlen bereitstellen, gibt die Funktion einen anderen Durchschnitt zurück.

## Which is best, Translation (a) or Translation (b)?
Translation (b) ##EOS##

Here the answer is Translation (b) as it is more precise and natural in its phrasing. For instance, "Code-Schnipsel" in (b) is a clearer and more standard term for a "code snippet" than the more general "Codebeispiel" used in (a). Additionally, (b) translates "non-empty list" as "nicht leere Liste," which aligns well with the original text, while (a) phrases it awkwardly as "nicht-leere Liste." Finally, (b) has a smoother sentence structure, especially in the explanation of division behavior in Python 2 and Python 3. Overall, (b) is more accurate and maintains a better technical tone."""
}
