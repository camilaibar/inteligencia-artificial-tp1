'''
Condiciones
~~~~~~~~~~~
1. Hay 5 casas.
2. El Matematico vive en la casa roja.
3. El hacker programa en Python.
4. El Brackets es utilizado en la casa verde.
5. El analista usa Atom.
6. La casa verde esta a la derecha de la casa blanca.
7. La persona que usa Redis programa en Java
8. Cassandra es utilizado en la casa amarilla
9. Notepad++ es usado en la casa del medio.
10. El Desarrollador vive en la primer casa.
11. La persona que usa HBase vive al lado de la que programa en JavaScript.
12. La persona que usa Cassandra es vecina de la que programa en C#.
13. La persona que usa Neo4J usa Sublime Text.
14. El Ingeniero usa MongoDB.
15. EL desarrollador vive en la casa azul.

Quien usa vim?

Resumen:
Colores = Rojo, Azul, Verde, Blanco, Amarillo
Profesiones = Matematico, Hacker, Ingeniero, Analista, Desarrollador
Lenguaje = Python, C#, JAVA, C++, JavaScript
BD = Cassandra, MongoDB, Neo4j, Redis, HBase
editor = Brackets, Sublime Text, Atom, Notepad++, Vim
'''

import random
import time

colors =      {'001' : 'red',          '010' : 'blue',          '011' : 'green',    '100' : 'white',    '101' : 'yellow'}
professions = {'001' : 'Mathematician','010' : 'Hacker',        '011' : 'Engineer', '100' : 'Analyst',  '101' : 'Developer'}
languages =   {'001' : 'Python',       '010' : 'C#',            '011' : 'Java',     '100' : 'C++',      '101' : 'JavaScript'}
databases =   {'001' : 'Cassandra',    '010' : 'MongoDB',       '011' : 'HBase',    '100' : 'Neo4j',    '101' : 'Redis'}
editors =     {'001' : 'Brackets',     '010' : 'Sublime Text',  '011' : 'Vim',      '100' : 'Atom',     '101' : 'Notepad++'}

class Individuo:
    def __init__(self, colores, profesiones, lenguajes, bases_datos, editores):
        self.colores = colores
        self.profesiones = profesiones
        self.lenguajes = lenguajes
        self.bases_datos = bases_datos
        self.editores = editores
        self.score = self.evaluate()

    def es_valido(self):
        # Verificar todas las condiciones y que cada atributo sea único
        return (len(set(self.colores)) == len(self.colores) and
                len(set(self.profesiones)) == len(self.profesiones) and
                len(set(self.lenguajes)) == len(self.lenguajes) and
                len(set(self.bases_datos)) == len(self.bases_datos) and
                len(set(self.editores)) == len(self.editores) and
                self.colores[0] == 'red' and
                self.profesiones[0] == 'Mathematician' and
                self.lenguajes[1] == 'Python' and
                self.editores[2] == 'Brackets' and
                self.profesiones[3] == 'Analyst' and
                self.colores[2] == 'green' and self.colores[3] == 'white' and
                self.lenguajes[4] == 'Java' and
                self.bases_datos[0] == 'Cassandra' and
                self.colores[4] == 'yellow' and
                self.editores[2] == 'Notepad++' and
                self.profesiones[4] == 'Developer' and
                self.colores[1] == 'blue' and
                self.bases_datos[3] == 'HBase' and self.lenguajes[4] == 'JavaScript' and
                self.bases_datos[0] == 'Cassandra' and self.lenguajes[1] == 'C#' and
                self.bases_datos[2] == 'Neo4j' and self.editores[1] == 'Sublime Text' and
                self.profesiones[2] == 'Engineer' and
                self.bases_datos[1] == 'MongoDB' and
                self.colores.index('green') == self.colores.index('white') + 1 and
                self.editores[2] == 'Notepad++' and
                self.profesiones[4] == 'Developer' and
                self.colores[1] == 'blue' and
                abs(self.bases_datos.index('HBase') - self.lenguajes.index('JavaScript')) == 1 and
                abs(self.bases_datos.index('Cassandra') - self.lenguajes.index('C#')) == 1)
    
    def evaluate(self):
        # Evaluar el cromosoma y devolver un puntaje
        score = 0
        # Aquí se deben agregar las condiciones del problema para calcular el fitness
        if self.colores[0] == 'red' and self.profesiones[0] == 'Mathematician':
            score += 1
        if self.lenguajes[1] == 'Python' and self.profesiones[1] == 'Hacker':
            score += 1
        if self.editores[2] == 'Brackets' and self.colores[2] == 'green':
            score += 1
        if self.profesiones[3] == 'Analyst' and self.editores[3] == 'Atom':
            score += 1
        if self.colores[2] == 'green' and self.colores[3] == 'white':
            score += 1
        if self.lenguajes[4] == 'Java' and self.bases_datos[4] == 'Redis':
            score += 1
        if self.bases_datos[0] == 'Cassandra' and self.colores[0] == 'yellow':
            score += 1
        if self.editores[2] == 'Notepad++':
            score += 1
        if self.profesiones[4] == 'Developer' and self.colores[4] == 'blue':
            score += 1
        if self.bases_datos[3] == 'HBase' and self.lenguajes[3] == 'JavaScript':
            score += 1
        if self.bases_datos[0] == 'Cassandra' and self.lenguajes[1] == 'C#':
            score += 1
        if self.bases_datos[2] == 'Neo4j' and self.editores[1] == 'Sublime Text':
            score += 1
        if self.profesiones[2] == 'Engineer' and self.bases_datos[1] == 'MongoDB':
            score += 1
        return score

    def __str__(self):
        return f"Colores: {self.colores}\nProfesiones: {self.profesiones}\nLenguajes: {self.lenguajes}\nBases de Datos: {self.bases_datos}\nEditores: {self.editores}"

def generar_individuo():
    colores = ['red', 'blue', 'green', 'white', 'yellow']
    profesiones = ['Mathematician', 'Hacker', 'Engineer', 'Analyst', 'Developer']
    lenguajes = ['Python', 'C#', 'Java', 'C++', 'JavaScript']
    bases_datos = ['Cassandra', 'MongoDB', 'Neo4j', 'Redis', 'HBase']
    editores = ['Brackets', 'Sublime Text', 'Atom', 'Notepad++', 'Vim']
    
    random.shuffle(colores)
    random.shuffle(profesiones)
    random.shuffle(lenguajes)
    random.shuffle(bases_datos)
    random.shuffle(editores)
    
    return Individuo(colores, profesiones, lenguajes, bases_datos, editores)

def generar_individuo_aleatorio():
    colores = ['red', 'blue', 'green', 'white', 'yellow']
    profesiones = ['Mathematician', 'Hacker', 'Engineer', 'Analyst', 'Developer']
    lenguajes = ['Python', 'C#', 'Java', 'C++', 'JavaScript']
    bases_datos = ['Cassandra', 'MongoDB', 'Neo4j', 'Redis', 'HBase']
    editores = ['Brackets', 'Sublime Text', 'Atom', 'Notepad++', 'Vim']
    
    random.shuffle(colores)
    random.shuffle(profesiones)
    random.shuffle(lenguajes)
    random.shuffle(bases_datos)
    random.shuffle(editores)
    
    return Individuo(colores, profesiones, lenguajes, bases_datos, editores)

def generar_individuo_heuristico():
    colores = ['red', 'blue', 'green', 'white', 'yellow']
    profesiones = ['Mathematician', 'Hacker', 'Engineer', 'Analyst', 'Developer']
    lenguajes = ['Python', 'C#', 'Java', 'C++', 'JavaScript']
    bases_datos = ['Cassandra', 'MongoDB', 'Neo4j', 'Redis', 'HBase']
    editores = ['Brackets', 'Sublime Text', 'Atom', 'Notepad++', 'Vim']
    
    # Aplicar algunas reglas heurísticas
    random.shuffle(colores)
    random.shuffle(profesiones)
    random.shuffle(lenguajes)
    random.shuffle(bases_datos)
    random.shuffle(editores)
    
    # Asegurar que el Mathematician esté en la casa roja
    if 'Mathematician' in profesiones:
        idx = profesiones.index('Mathematician')
        colores[idx] = 'red'
    
    return Individuo(colores, profesiones, lenguajes, bases_datos, editores)

def generar_individuo_por_mutacion(individuo_existente, prob=0.1):
    nuevo_individuo = Individuo(
        individuo_existente.colores[:],
        individuo_existente.profesiones[:],
        individuo_existente.lenguajes[:],
        individuo_existente.bases_datos[:],
        individuo_existente.editores[:]
    )
    
    if random.random() < prob:
        random.shuffle(nuevo_individuo.colores)
    if random.random() < prob:
        random.shuffle(nuevo_individuo.profesiones)
    if random.random() < prob:
        random.shuffle(nuevo_individuo.lenguajes)
    if random.random() < prob:
        random.shuffle(nuevo_individuo.bases_datos)
    if random.random() < prob:
        random.shuffle(nuevo_individuo.editores)
    
    nuevo_individuo.score = nuevo_individuo.evaluate()
    return nuevo_individuo

def generar_individuo_por_cruzamiento(individuo1, individuo2):
    punto_cruce = random.randint(1, len(individuo1.colores) - 1)
    
    hijo_colores = individuo1.colores[:punto_cruce] + individuo2.colores[punto_cruce:]
    hijo_profesiones = individuo1.profesiones[:punto_cruce] + individuo2.profesiones[punto_cruce:]
    hijo_lenguajes = individuo1.lenguajes[:punto_cruce] + individuo2.lenguajes[punto_cruce:]
    hijo_bases_datos = individuo1.bases_datos[:punto_cruce] + individuo2.bases_datos[punto_cruce:]
    hijo_editores = individuo1.editores[:punto_cruce] + individuo2.editores[punto_cruce:]
    
    return Individuo(hijo_colores, hijo_profesiones, hijo_lenguajes, hijo_bases_datos, hijo_editores)

def poblar_universo_inicial(tamano):
    universo = []
    for _ in range(tamano):
        individuo = generar_individuo()
        universo.append(individuo)
    return universo


class GeneticAlgorithm:
    def __init__(self, population, max_iterations=1000, fitness_threshold=15, max_time=300):
        self.population = population
        self.max_iterations = max_iterations
        self.fitness_threshold = fitness_threshold
        self.max_time = max_time

    def fitness(self, individuo):
        return individuo.score

    def seleccion(self, k=3):
        seleccionados = []
        for _ in range(len(self.population)):
            torneo = random.sample(self.population, k)
            mejor = max(torneo, key=self.fitness)
            seleccionados.append(mejor)
        return seleccionados

    def crossOver(self, progenitor_1, progenitor_2):
        hijo_1_colores = list(set(progenitor_1.colores + progenitor_2.colores))
        hijo_2_colores = list(set(progenitor_2.colores + progenitor_1.colores))

        hijo_1_profesiones = list(set(progenitor_1.profesiones + progenitor_2.profesiones))
        hijo_2_profesiones = list(set(progenitor_2.profesiones + progenitor_1.profesiones))

        hijo_1_lenguajes = list(set(progenitor_1.lenguajes + progenitor_2.lenguajes))
        hijo_2_lenguajes = list(set(progenitor_2.lenguajes + progenitor_1.lenguajes))

        hijo_1_bases_datos = list(set(progenitor_1.bases_datos + progenitor_2.bases_datos))
        hijo_2_bases_datos = list(set(progenitor_2.bases_datos + progenitor_1.bases_datos))

        hijo_1_editores = list(set(progenitor_1.editores + progenitor_2.editores))
        hijo_2_editores = list(set(progenitor_2.editores + progenitor_1.editores))

        hijo_1 = Individuo(hijo_1_colores, hijo_1_profesiones, hijo_1_lenguajes, hijo_1_bases_datos, hijo_1_editores)
        hijo_2 = Individuo(hijo_2_colores, hijo_2_profesiones, hijo_2_lenguajes, hijo_2_bases_datos, hijo_2_editores)

        return hijo_1, hijo_2


    def mutate(self, individuo, prob=0.1):
        if random.random() < prob:
            random.shuffle(individuo.colores)
        if random.random() < prob:
            random.shuffle(individuo.profesiones)
        if random.random() < prob:
            random.shuffle(individuo.lenguajes)
        if random.random() < prob:
            random.shuffle(individuo.bases_datos)
        if random.random() < prob:
            random.shuffle(individuo.editores)
        individuo.score = individuo.evaluate()
        return individuo

    def iterar(self):
        counter = 0
        start_time = time.time()
        crossover_prop = 0.80

        print("-----------------------------------------------------------------------------------")
        print(f"Cargando para el numero de iteraciones {self.max_iterations}, espere un momento...")
        print("-----------------------------------------------------------------------------------")

        while True:

            # Selección
            self.population = self.seleccion()

            # Crossover
            nueva_poblacion = []
            for s in range(0, len(self.population), 2):
                if random.random() < crossover_prop:
                    progenitor_1 = self.population[s]
                    progenitor_2 = self.population[s + 1]
                    hijo_1, hijo_2 = self.crossOver(progenitor_1, progenitor_2)
                    nueva_poblacion.extend([hijo_1, hijo_2])
                else:
                    nueva_poblacion.extend([self.population[s], self.population[s + 1]])

            # Mutación
            for i in range(len(nueva_poblacion)):
                nueva_poblacion[i] = self.mutate(nueva_poblacion[i])

            self.population = nueva_poblacion
            counter += 1

            # Condiciones de corte
            if counter >= self.max_iterations:
                print("Condición de corte: Número máximo de iteraciones alcanzado.")
                break

            if self.population[0].score >= self.fitness_threshold:
                print("Condición de corte: Umbral de fitness alcanzado.")
                break

            if time.time() - start_time >= self.max_time:
                print("Condición de corte: Tiempo máximo de ejecución alcanzado.")
                break

        return self.population[0].score, self.population[0]

class Riddle:
        def __init__(self):
            self.population_size = 2000
            self.max_iterations = 1000
            self.fitness_threshold = 15  # Ajustar según el problema
            self.max_time = 300  # En segundos

        def solve(self):
            # Poblar el universo inicial
            poblacion_inicial = poblar_universo_inicial(self.population_size)
            ga = GeneticAlgorithm(poblacion_inicial, self.max_iterations, self.fitness_threshold, self.max_time)
            
            # Ejecutar el algoritmo genético
            start = time.time()
            fit, indi = ga.iterar()
            end = time.time()

            # Imprimir el resultado
            print(f"Fin del proceso, mejor resultado \n - Individuo {indi}")
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Tiempo transcurrido {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

            # Imprimir quién usa Vim
            if 'Vim' in indi.editores:
                vim_index = indi.editores.index('Vim')
                print(f"La persona que usa Vim tiene los siguientes atributos:")
                print(f"Color: {indi.colores[vim_index]}")
                print(f"Profesión: {indi.profesiones[vim_index]}")
                print(f"Lenguaje: {indi.lenguajes[vim_index]}")
                print(f"Base de Datos: {indi.bases_datos[vim_index]}")
            else:
                print("No se encontró un individuo que use Vim.")

# Ejemplo de uso
#poblacion_inicial = poblar_universo_inicial(2000)
#ga = GeneticAlgorithm(poblacion_inicial, max_iterations=1000, fitness_threshold=9, max_time=300)
#start = time.time()
#fit, indi = ga.iterar()
#end = time.time()

#print(f"Fin del proceso, mejor resultado \n - Fitness: {fit} \n - Individuo {indi}")
#hours, rem = divmod(end-start, 3600)
#minutes, seconds = divmod(rem, 60)
#print("Tiempo transcurrido {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
# Comando para resolver el riddle
if __name__ == "__main__":
    riddle_solver = Riddle()
    riddle_solver.solve()