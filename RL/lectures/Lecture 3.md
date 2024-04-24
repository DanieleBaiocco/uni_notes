# Finite Markov Decision Process
Mi serve sapere le dynamics del sistema per definire un MDP, quindi devo essere a conscienza del modello praticamente. Finite perche' il numero di stati  e' finito e posso vederlo (lo stesso per le azioni). Ovviamente l'MDP deve rispettare la Markov Property, quindi lo stato attuale deve essere abbastanza per poter dire le dinamiche dell'enviroment, ovvero come questo si evolvera' al prossimo step (in che stato andra' e con che reward) alla selezione di un'azione:
![[dymanics_mdp.PNG]]
tra l'altro ho una distribuzione di probabilita' dato s e a, per cui gli stati in cui andro' con un determinato reward hanno una probabilita' e se le sommo fanno 1.
![[dynamics_mdp_2.PNG]]
Il return di un'episodio e':
![[return_mdp.PNG]]
Tra l'altro ho anche sto risultato bellissimo sul return:
![[return_mdp_2.PNG]]
A me serve calcolare l'**expected return** per ogni stato perche' in questo modo so quanto e' buono stare in un determinato stato (seguendo una policy fissata ovviamente). Quanto reward mi dara' di media stare sullo stato in cui sto seguendo una determinata policy? Questo si chiama lo state-value function:
![[value_function_simple.PNG]]
E  se considero che ho gia fatto un'azione, quanto e' buono lo stato in cui vado? Questo e' l'action-value function:
![[action_value_simple.PNG]]
La formula sbobinata per lo state-value function, in cui posso esprimerlo in funzione di se stesso e' la seguente:
![[state_value_extended.PNG]]


Comunque questa rappersentazione a grafo dell'MDP e' tipo importantissima quindi guardala:
![[esempio_MDP.PNG]]