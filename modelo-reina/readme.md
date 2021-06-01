
https://github.com/kausaltech/reina-model

### Contactos

El número de contactos diarios por persona se obtiene a partir de la edad y de la matriz de contactos, mediante una distribución *log-normal*:

    context.random.lognormal(0, 0.5) * self.nr_contacts_by_age[person.age]

El número máximo de contactos que puede tener una persona es ```128``` o el definido mediante la intervención ```limit_mass_gatherings```.
Una vez calculado el número de contactos, la asignación de contactos para cada agente se realiza teniendo en cuenta la matriz de contactos, y se mantiene toda la simulación.



### Interventions

- ```limit-mass-gatherings```: limita el número máximo de contactos que puede establecer una persona cada día. El valor ```0``` no tiene efecto. 
- 