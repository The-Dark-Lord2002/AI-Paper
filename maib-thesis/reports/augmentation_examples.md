# Data augmentation: original vs rewritten reports

Language model: `Qwen/Qwen2.5-1.5B-Instruct`. Target per class: 400 training reports. Label check: on.

| Class | Real | Rewritten | Passed label check | Used for training |
|---|---:|---:|---:|---:|
| Accident to person(s) | 1163 | 0 | 0 | 1163 |
| Damage / Loss Of Equipment | 962 | 0 | 0 | 962 |
| Loss Of Control | 576 | 0 | 0 | 576 |
| Grounding / Stranding | 393 | 12 | 7 | 400 |
| Contact | 284 | 215 | 116 | 400 |
| Collision | 228 | 187 | 172 | 400 |
| Fire / Explosion | 227 | 184 | 173 | 400 |
| Flooding / Foundering | 131 | 408 | 269 | 400 |
| Capsizing / Listing | 69 | 487 | 331 | 400 |

Overall 1068 of 1493 rewrites (71.5%) passed the label check.

## Capsizing / Listing

**Original:** A rescue boat capsized and grounded whilst rescuing some people cut off by the tide. A search and rescue vessel was responding to two people and a dog cut off by the tide when they lost control of the vessel due to the severity of the swell. They beached the vessel in order to access the beach to rescue the trapped people but were not able to return to the vessel due to the swell. They secured the vessel by anchor and the 3 crew of the SAR vessel were taken up the cliff by the shore side rescue team along with the two people and the dog. The abandoned SAR vessel was left overnight during which time it capsized due to the waves. This resulted in minor damage to the engine, lighting rig and the trim. As a precaution the vessel was towed away and taken to a shore based repair location to check and overhaul as necessary, but no severe or disabling damage was reported.

**Rewrite:** A rescue boat capsized while rescuing individuals who had been cut off by rising waters. The SAR vessel, responding to an emergency involving two individuals and a canine, suffered from the forceful surge of the sea. In their attempt to reach the coast, they became overwhelmed by the stormy conditions and found themselves unable to regain control of their vessel. Unable to return to the stranded individuals on land, they anchored the craft for safety purposes. However, their efforts did not prevent a subsequent incident; the vessel began to capsize after being caught by large waves.

---

**Original:** A sailing vessel capsized, two crew fell into the water and were recovered by a search and rescue vessel A sailing vessel with two crew onboard capsized, the crew were unable to right the vessel and were seen hanging on. The Coastguard were informed and tasked a search and rescue (SAR) vessel to the scene. The SAR vessel recovered one crew member and a nearby vessel had recovered the other who was then transferred to the SAR vessel. They were both assessed by SAR crew and taken ashore. The capsized vessel drifted to shore and was recovered.

**Rewrite:** A sailing vessel capsized, causing the loss of two crew members who floated in the water until rescued by a nearby search and rescue vessel. Upon being alerted, the coast guard dispatched a rescue unit to the area where they located the stricken vessel. A sailor from the local ship managed to assist in recovering one of the crew, while another crew member was subsequently saved by a passing ferry and handed over for further care. Both victims were evaluated at the station, with one being discharged after receiving medical attention, while the second was taken aboard the rescue vessel. The capsized vessel later drifted ashore and was successfully retrieved by the salvage team.

---

**Original:** A tender capsized with two crew members entering the water, both were recovered A sail training vessel's crew members were recovering the tender after departing a port, the two crew stepped out onto ladder and the tender capsized and the two crew entering the water. The crew righted the tender and were recovered. No injuries sustained.

**Rewrite:** Two crew members on a tender fell into the water while helping another vessel depart from a harbor. After righting the capsized tender, they were safely recovered without any injuries.

---

**Original:** A search and rescue vessel capsized in the surf resulting in two crew in the water A search and rescue (SAR) vessel with two crew onboard was launching on exercise. The vessel was hit by a wave on the shore edge and capsized. Both the vessel and crew washed up on the shore. Shore crew assisted to lift the vessel off the crew. Both crew were taken to hospital for suspected water inhalation but were released with no further injuries.

**Rewrite, REJECTED by the label check:** A SAR vessel collided with a wave along the shoreline during an exercise launch. The capsizing led to both the ship and its crew being washed onto the shore. Coastguards successfully retrieved the vessel using shore equipment. Fortunately, there were no significant injuries among the injured personnel as they were quickly transferred to medical facilities for examination.

---

## Collision

**Original:** A search and rescue vessel collided with a recreational craft during a tasking. A search and rescue vessel collided with a pleasure craft during an emergency tasking. One of the crew members for the search and rescue vessel was on the pleasure craft which was moored at the time. The skipper was attempting to pick them up so that the vessel could respond to reports of a person in the water. The skipper of the search and rescue vessel was navigating at slow speed when they lost control and hit the pleasure craft. This resulted in minor damage to the bow of the search and rescue vessel and the gunwale of the pleasure craft. There were no injuries or pollution reported. The incident was attributed to an error on the part of the skipper of the search and rescue vessel as they approached the other vessel.

**Rewrite:** During an emergency operation, a search and rescue vessel encountered a recreational craft while trying to assist. One member of the search and rescue team from one of the vessels was also onboard the pleasure craft which had been anchored. The skipper aimed to retrieve the personnel aboard the pleasure boat; however, the slow-moving search and rescue vessel veered off course and struck the pleasure craft. This collision caused slight damage to the bow section of the search and rescue vessel and the edge of the pleasure craft. No injuries or environmental concerns were noted. The accident stemmed from a navigational oversight by the captain of the search and rescue vessel, who was approaching the other vessel too closely without sufficient caution.

---

**Original:** A tug collided with a crude oil tanker resulting in minor during berthing operations. A tug was assisting a crude oil tanker with departure manoeuvring when it made contact with the shell plating of the port quarter of the tanker. This allegedly resulted in denting to the shell plating of the tanker. There was no damage reported to the tug. The crude oil tanker was able to depart and continue on passage without further issue. There was no pollution or injuries.

**Rewrite:** A tug collided with a crude oil tanker during berthing operations, causing minor damage. The tug was assisting the crude oil tanker as it departed when it struck the hull's port quarter, resulting in dents to the tankers shell plating. No damage was reported to the tug. Despite the collision, the crude oil tanker was able to depart and continue its journey unhindered. There were no incidents involving pollution or injuries.

---

**Original:** Near miss occurred between a fishing vessel and a yacht. Yacht was proceeding towards a marina when a fishing vessel was coming out of harbour. There was light seen in the fishing vessel's wheelhouse. The yacht moved to starboard to give space to the fishing boat however, the fishing boat continued to port side of the yacht. The yacht immediately turned turn to port to avoid collision and passed the fishing vessel to starboard. The fishing vessel corrected their course and continued their passage. The crew from the fishing vessel reported that they have not seen the yacht. No VHF messages were passed or sound signals made. The fishing vessel was not using AIS.

**Rewrite:** A near collision involving a fishing vessel and a yacht took place. The yacht was heading towards a marina at high speed while a fishing vessel exited the harbor nearby. At this time, a figure could be spotted through the steering mechanism of the fishing vessel. Despite attempting to maintain distance by turning to the right, the fishing vessel approached the yacht on its left side. Panicked, the yacht captain immediately steered his ship to the left to prevent an imminent accident. As a result, he managed to pass the fishing vessel on the starboard side before it could catch up. However, the fishing vessel did not alter course and continued downstream. Reports indicated that there were no instances of VHF radio communication or audible alerts exchanged between the vessels. Furthermore, the fishing vessel lacked installed Automatic Identification System (AIS) technology.

---

**Original:** Not available Under Permanent Cooperation Framework (PCF) Not available Under Permanent Cooperation Framework (PCF)

**Rewrite, REJECTED by the label check:** No information provided under a permanent cooperation framework for this incident.

---

## Contact

**Original:** High speed passenger vessel made contact with an unidentified object resulting in propeller damage. A high speed passenger vessel sustained two impacts whilst inbound and there was vibration on P.M.E. On arrival, all compartments were checked visually and all seemed fine. However, there was damage to the port propeller blades resulting in the vessel being taken out of service for repair.

**Rewrite:** A high-speed passenger vessel collided with an unknown object and experienced propeller damage. The ship encountered two collisions while proceeding towards its destination, causing vibrations at the propulsion equipment monitoring (P.M.E.). Upon arrival, visual inspections of all compartments revealed no issues; however, significant damage to the port propeller blades necessitated the vessel's removal from service for repairs.

---

**Original:** A crew transfer vessel made contact with a turbine which resulted in minor damage to vessel. A crew transfer vessel manoeuvred away from a turbine against the wind when it started to drift. The vessel was getting closer to the turbine and the stern portside platform made contact with it. No damage to the turbine reported however, slight damage was reported to the fender bar above the water jets.

**Rewrite:** The crew transfer vessel collided with a turbine during maneuvering operations against the wind. It approached too closely and the stern portside platform struck it. There was no significant damage to the turbine but there were slight damages noted on the fender bar above the water jets.

---

**Original:** A windfarm vessel made contact with a wind turbine A windfarm vessel was transferring personnel to windfarm locations in the area. With two passengers still on board, the master left the bridge and handed over the helm to the deckhand, the vessel was in neutral. The bilge alarm on the port engine sounded, the master went to check the engine and observed an oil leak that likely caused the alarm. It was noticed that the vessel was creeping slowly astern towards a turbine. This was a characteristic of the propulsion system when not disengaged from the drive line. The vessel made contact with the turbine causing damage to the starboard jet platform and bucket. The master returned, took control and manoeuvred clear. No pollution seen or injuries.

**Rewrite:** A vessel transporting personnel to wind farm sites collided with a wind turbine during its transit. Two crew members remained aboard while the captain disembarked onto the bridge and entrusted the helm to the deck officer. As the vessel was operating in neutral gear, the bilge alarm for the port engine activated, prompting the captain to investigate the problem. Upon inspection, he discovered an oil leakage which triggered the alarm. Observing that the vessel was drifting astern, indicative of improper engagement with the propeller drive system, the ship approached a wind turbine. This behavior was typical due to unsecured propulsion settings. Unfortunately, the vessel's collision with the turbine resulted in structural damage to the starboard jet platform and buckets. To rectify the situation, the captain retook control and maneuvered away safely. The incident did not result in any environmental contamination or harm to the crew.

---

**Original:** A general cargo ship made contact with a concrete quayside and sustained material damage. A general cargo ship made contact with a concrete quayside and sustained material damage.

**Rewrite, REJECTED by the label check:** A general cargo ship collided with a concrete quay wall and experienced structural damage.

---

## Fire / Explosion

**Original:** A cruise ship experienced an explosion an air conditioning unit in the engine room and subsequent loss of propulsion A cruise ship experienced an explosion on an air-conditioning unit ultimately leading to the failure of propulsion systems for approximately 20 minutes. The explosion of the air-conditioning unit triggered a fire alarm which activated the water mist system for the adjacent diesel generator. The emergency stop for the diesel generator was activated as a precaution in case it was the unit on fire. The two remaining diesel generators on load were maintained throughout. The effect of the explosion on the air-conditioning unit resulted in a frequency disruption and voltage drop on the main switchboard. This lead to the variable frequency drives for the azipod cooling fans being disrupted and causing the azipods to trip. The propulsion was restored after 20 minutes. Steering capability was maintained throughout the incident.

**Rewrite:** The cruise ship had an explosion involving an air conditioning unit within its engine room, resulting in temporary loss of propulsion. An explosion at this component set off the activation of the nearby diesel generator's water mist system due to the fire alarm triggered by the initial event. Emergency stops were initiated for any potential fires related to the malfunctioning unit. Although both available generators were operational, one remained running while the other provided backup power. The explosion affected the air conditioning unit, disrupting the system's frequency supply and consequently tripping the variable frequency drives for the propeller pumps. Following restoration from the initial state, steering capabilities were unaffected throughout the incident period.

---

**Original:** A hopper dredger had a small engine fire after an oil leak. A hopper dredger had a small engine fire after the starboard engine gearbox blew out spraying the engine room in oil. The fire was extinguished using a foam trolley extinguisher. The vessel anchored overnight and arranged to head to port for repair. No persons were injured.

**Rewrite:** A hopper dredger experienced a minor engine fire due to an oil leak from the starboard engine gearbox. The gearbox ruptured, causing oil to spray into the engine room. Fire-fighting efforts used a foam trolley extinguisher. The ship was secured during the night before heading to port for repairs. Fortunately, no personnel were harmed.

---

**Original:** Ro-Pax main engine fire resulting returning to port as repairs were unsuccessful. Ro-Pax main engine fire resulting returning to port as repairs were unsuccessful.

**Rewrite:** Engine fire on Ro-Pax vessel forcing return to port due to repair failure.

---

**Original:** Sailing vessel crewmember suffered burns from a gas explosion after cooking using LPG. A sailing vessel crewmember sustained facial and leg burns from a gas explosion as a result of cooking using LPG. Multiple assets were tasked with the casualty being extracted from the vessel and taken to hospital.

**Rewrite, REJECTED by the label check:** A sailing vessel crew member experienced burns due to a gas explosion while cooking using liquefied petroleum gas (LPG). The injury resulted in facial and leg burns for the employee. Rescue efforts involved evacuating the casualty from the vessel and transporting them to the nearest medical facility.

---

## Flooding / Foundering

**Original:** A motor vessel sunk at its berth with one person on board who evacuated to the quayside. A motor cruiser with one person on board was alongside a quay wall and started taking on water. The crewmember was able to disembark safely before the vessel flooded and sat on the seabed. The source of the flooding was thought to be the exhaust bellows. There was some pollution from the fuel tank into the harbour.

**Rewrite:** At its berth, a motor vessel sank, leaving one person aboard who eventually evacuated to the quayside. While another motor cruiser was alongside a quay wall, it began to take in water due to a leaky exhaust bellows. The crew managed to safely exit the boat prior to flooding, causing it to sit on the seabed. It is believed that the fuel tank leaked into the harbor as a result of this incident.

---

**Original:** A recreational motor vessel had water ingress and required assistance. A recreational motor vessel had water ingress and was drifting. The crew onboard were able to tie to a mooring buoy and started to bail out the water but were unable to keep up with the ingress. A passing motor vessel offered assistance whilst a search and rescue vessel was deployed. The crew were taken ashore and the stricken vessel towed to safety.

**Rewrite:** A recreational motor vessel experienced water ingress and drifted. The crew managed to secure themselves on a mooring buoy using ropes and began bailing out the water, but the ingress continued at an increasing rate. An approaching motor vessel provided assistance while a search and rescue team dispatched a vessel for further aid. Crew members were safely evacuated, and the distressed vessel was towed to safety by another ship.

---

**Original:** A recreational motor vessel with 2 POB sank with one loss of life A recreational motor vessel foundered with two persons on board. Both were elderly, one lost their life. They were both recovered from the water by local coastguard services. The surviving person was assessed by the ambulance before being released. The other person was declared dead after efforts to revive them were unsuccessful. The circumstances of the sinking remain uncertain and the vessel has not been raised from where it sank, nor has the exact position of the wreck been determined. Foundering of a recreational vessel with two persons on board. The occupants were recovered from the water but one was subsequently declared deceased.

**Rewrite:** A recreational boat carrying two people sank in rough seas. Both individuals perished while they were out on the water. Their bodies were rescued by the local coast guard and taken to shore for medical attention. Unfortunately, the second survivor could not be revived despite all attempts to save him or her. Despite these tragic events, the cause of the sinking remains unknown as no wreckage has been located, nor can the specific location of the submerged boat be identified. A recreational vessel sank with two passengers onboard due to poor weather conditions. Two people died during this maritime accident; however, they were successfully retrieved by the coast guard. Unfortunately, the remaining individual did not survive and passed away following rescue operations.

---

**Original:** High speed passenger ferry suffered an engine room flood An inland waters high speed foot passenger ferry experienced an engine room flood. The passengers were disembarked from the vessel. Local emergency services attended to pump out the water from the engine room and then the vessel was escorted back to a pier. There were no reported injuries or pollution. The source of the flooding was not reported.

**Rewrite, REJECTED by the label check:** A fast passenger ship encountered water in its engine compartment. Passengers were removed from the boat before it was towed to shore due to the flooding. Emergency responders managed to remove the water and returned the vessel safely. No injuries occurred nor did any pollutants spill. The reason for the engine room's failure is unknown.

---

## Grounding / Stranding

**Original:** A sailing vessel ran aground and was taking on water. A catamaran yacht ran aground on a rock and was holed in the right hull. The engine then failed as a result. The crew issued a MAYDAY call and a lifeboat provided pumps and a tow to a local boatyard for repairs.

**Rewrite:** A sailing vessel came upon trouble by running aground and taking on water. A catamaran yacht encountered the same issue when it hit rocks and got stuck with its right hull breached. Due to mechanical failure, the power stopped working. Crew members sounded an emergency signal, requesting assistance. An inflatable life raft arrived to assist, pumping out excess water and providing a tow to bring the damaged vessel to shore for repairs at a nearby marina.

---

**Original:** A motor cruiser grounded, sustaining propeller damage, and had to be refloated and towed by a lifeboat. A motorboat with two anglers on board grounded on a beach, it appears intentionally. The crew went ashore for food. On trying to depart from the beach on the rising tide it was evident the vessel's propeller was damaged and the vessel had no other means of propulsion. A lifeboat successfully re-floated the vessel and a tow was established to return the vessel to its home port.

**Rewrite:** A motor cruiser ran aground, resulting in propeller damage. It required refloating and towing by a lifeboat. While a motorboat with two anglers aboard experienced an intentional grounding on a beach, they were able to go ashore for sustenance. Upon attempting departure during high tide, the vessel's propeller was damaged, leaving it without propulsion. A successful lifeboat operation managed to refloat the vessel and establish a tow to transport it back to its home port.

---

**Original:** A commercial RIB grounded on a sandbar and was towed by a privately owned vessel. A RIB (rigid inflatable boat) grounded on a sandbar and another RIB towed them off safely and back into port. No damage was confirmed.

**Rewrite:** A commercial Rigid Inflatable Boat (RIB) ran aground on a sandbar and was later rescued by a privately owned vessel. The additional RIBs were successfully towed to safety for eventual safe return to port without any reported damage.

---

**Original:** A SAR vessel grounded whilst exiting the harbour. A search and rescue (SAR) vessel grounded whilst exiting the harbour. The harbour is subject to a sand bar building at the entrance after bad weather. The vessel was still able to leave the harbour and safety checks were carried out to ensure vessel was undamaged.

**Rewrite, REJECTED by the label check:** A SAR vessel got stuck while leaving the harbor due to the harbor being affected by a growing sandbar after rough weather. Although the vessel managed to get out of the harbor, it underwent safety inspections to confirm its condition.

---
