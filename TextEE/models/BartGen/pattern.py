patterns = {
    "ace05-en": {
        "Business:Declare-Bankruptcy": {
            "template": "<arg1> declared bankruptcy at <arg2> place",
            "roles": ["Org", "Place"],
        },
        "Business:End-Org": {
            "template": "<arg1> organization shut down at <arg2> place",
            "roles": ["Org", "Place"],
        },
        "Business:Merge-Org": {
            "template": "<arg1> organization merged",
            "roles": ["Org"],
        },
        "Business:Start-Org": {
            "template": "<arg1> started <arg2> organization at <arg3> place",
            "roles": ["Agent", "Org", "Place"],
        },
        "Conflict:Attack": {
            "template": "<arg1> attacked <arg2> hurting <arg5> victims using <arg3> instrument at <arg4> place",
            "roles": ["Attacker", "Target", "Instrument", "Place", "Victim"],
        },
        "Conflict:Demonstrate": {
            "template": "<arg1> demonstrated at <arg2> place",
            "roles": ["Entity", "Place"],
        },
        "Contact:Meet": {
            "template": "<arg1> met with <arg2> in <arg3> place",
            "roles": ["Entity", "Entity", "Place"],
        },
        "Contact:Phone-Write": {
            "template": "<arg1> communicated remotely with <arg2> at <arg3> place",
            "roles": ["Entity", "Entity", "Place"],
        },
        "Justice:Acquit": {
            "template": "<arg1> court or judge acquitted <arg2>",
            "roles": ["Adjudicator", "Defendant"],
        },
        "Justice:Appeal": {
            "template": "<arg1> court or judge acquitted <arg2>",
            "roles": ["Adjudicator", "Defendant"],
        },
        "Justice:Arrest-Jail": {
            "template": "<arg1> arrested <arg2> in <arg3> place",
            "roles": ["Agent", "Person", "Place"],
        },
        "Justice:Charge-Indict": {
            "template": "<arg1> charged or indicted <arg2> before <arg3> court or judge in <arg4> place",
            "roles": ["Prosecutor", "Defendant", "Adjudicator", "Place"],
        }, 
        "Justice:Convict": {
            "template": "<arg1> court or judge convicted <arg2> in <arg3> place",
            "roles": ["Adjudicator", "Defendant", "Place"],
        }, 
        "Justice:Execute": {
            "template": "<arg1> executed <arg2> at <arg3> place",
            "roles": ["Agent", "Person", "Place"],
        },
        "Justice:Extradite": {
            "template": "<arg1> extradited <arg2> from <arg3> place to <arg4> place",
            "roles": ["Agent", "Person", "Origin", "Destination"],
        },
        "Justice:Fine": {
            "template": "<arg1> court or judge fined <arg2> at <arg3> place",
            "roles": ["Adjudicator", "Entity", "Place"],
        },
        "Justice:Pardon": {
            "template": "<arg1> court or judge pardoned <arg2> at <arg3> place",
            "roles": ["Adjudicator", "Defendant", "Place"],
        }, 
        "Justice:Release-Parole": {
            "template": "<arg1> released or paroled <arg2> in <arg3> place",
            "roles": ["Entity", "Person", "Place"],
        },
        "Justice:Sentence": {
            "template": "<arg1> court or judge sentenced <arg2> in <arg3> place",
            "roles": ["Adjudicator", "Defendant", "Place"],
        }, 
        "Justice:Sue": {
            "template": "<arg1> sued <arg2> before <arg3> court or judge in <arg4> place",
            "roles": ["Plaintiff", "Defendant", "Adjudicator", "Place"],
        }, 
        "Justice:Trial-Hearing": {
            "template": "<arg1> tried <arg2> before <arg3> court or judge in <arg4> place",
            "roles": ["Prosecutor", "Defendant", "Adjudicator", "Place"],
        }, 
        "Life:Be-Born": {
            "template": "<arg1> was born in <arg2> place",
            "roles": ["Person", "Place"],
        }, 
        "Life:Die": {
            "template": "<arg1> killed <arg2> with <arg3> instrument in <arg4> place",
            "roles": ["Agent", "Victim", "Instrument", "Place"],
        },
        "Life:Divorce": {
            "template": "<arg1> divorced <arg2> in <arg3> place",
            "roles": ["Person", "Person", "Place"],
        }, 
        "Life:Injure": {
            "template": "<arg1> injured <arg2> with <arg3> instrument in <arg4> place",
            "roles": ["Agent", "Victim", "Instrument", "Place"],
        },
        "Life:Marry": {
            "template": "<arg1> married <arg2> in <arg3> place",
            "roles": ["Person", "Person", "Place"],
        },
        "Movement:Transport": {
            "template": "<arg1> transported <arg2> in <arg3> vehicle from <arg4> place to <arg5> place",
            "roles": ["Agent", "Artifact", "Vehicle", "Origin", "Destination"],
        },
        "Personnel:Elect": {
            "template": "<arg1> elected <arg2> in <arg3> place",
            "roles": ["Entity", "Person", "Place"],
        },
        "Personnel:End-Position": {
            "template": "<arg1> stopped working at <arg2> organization in <arg3> place",
            "roles": ["Person", "Entity", "Place"],
        }, 
        "Personnel:Nominate": {
            "template": "<arg1> nominated <arg2>",
            "roles": ["Agent", "Person"],
        },  
        "Personnel:Start-Position": {
            "template": "<arg1> started working at <arg2> organization in <arg3> place",
            "roles": ["Person", "Entity", "Place"],
        },  
        "Transaction:Transfer-Money": {
            "template": "<arg1> gave money to <arg2> for the benefit of <arg3> in <arg4> place",
            "roles": ["Giver", "Recipient", "Beneficiary", "Place"],
        },  
        "Transaction:Transfer-Ownership": {
            "template": "<arg1> gave <arg4> to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ["Seller", "Buyer", "Beneficiary", "Artifact", "Place"],
        }, 
    }, 
    "richere-en": {
        "Business:Declare-Bankruptcy": {
            "template": "<arg1> declared bankruptcy",
            "roles": ['Org'],
        },
        "Business:End-Org": {
            "template": "<arg1> dissolved in <arg2> place",
            "roles": ['Org', 'Place'],
        },
        "Business:Merge-Org": {
            "template": "<arg1> was merged",
            "roles": ['Org'],
        },
        "Business:Start-Org": {
            "template": "<arg1> launched <arg2> at <arg3> place",
            "roles": ['Agent', 'Org', 'Place'],
        },
        "Conflict:Attack": {
            "template": "<arg1> attacked <arg2> by <arg3> at <arg4> place",
            "roles": ['Attacker', 'Target', 'Instrument', 'Place', 'Agent'],
        },
        "Conflict:Demonstrate": {
            "template": "<arg1> protested at <arg2> place",
            "roles": ['Entity', 'Place'],
        },
        "Contact:Broadcast": {
            "template": "<arg1> made announcement to <arg2> at <arg3> place",
            "roles": ['Entity', 'Audience', 'Place'],
        },
        "Contact:Contact": {
            "template": "<arg1> talked to each other at <arg2> place",
            "roles": ['Entity', 'Place'],
        },
        "Contact:Correspondence": {
            "template": "<arg1> contacted each other at <arg2> place",
            "roles": ['Entity', 'Place'],
        },
        "Contact:Meet": {
            "template": "<arg1> met at <arg2> place",
            "roles": ['Entity', 'Place'],
        },
        "Justice:Acquit": {
            "template": "<arg1> was acquitted of the charges by <arg2> at <arg3> place",
            "roles": ['Defendant', 'Adjudicator', 'Place'],
        },
        "Justice:Appeal": {
            "template": "<arg1> in <arg2> place appealed the adjudication by <arg4> from <arg3> adjudicator",
            "roles": ['Defendant', 'Place', 'Adjudicator', 'Prosecutor'],
        },
        "Justice:Arrest-Jail": {
            "template": "<arg1> was sent to jailed or arrested by <arg2> at <arg3> place",
            "roles": ['Person', 'Agent', 'Place'],
        },
        "Justice:Charge-Indict": {
            "template": "<arg1> was charged by <arg2> at <arg3> place, and the adjudication was judged by <arg4> adjudicator",
            "roles": ['Defendant', 'Prosecutor', 'Place', 'Adjudicator'],
        },
        "Justice:Convict": {
            "template": "<arg1> was convicted of a crime in <arg2> place, and the adjudication was judged by <arg3> adjudicator",
            "roles": ['Defendant', 'Place', 'Adjudicator'],
        },
        "Justice:Execute": {
            "template": "<arg1> was executed by <arg2> at <arg3> place",
            "roles": ['Person', 'Agent', 'Place'],
        },
        "Justice:Extradite": {
            "template": "<arg1> was extradicted to <arg2> from <arg3>, and <arg4> was responsible for the extradition",
            "roles": ['Person', 'Destination', 'Origin', 'Agent'],
        },
        "Justice:Fine": {
            "template": "<arg1> in <arg2> place was ordered by <arg3> to pay a fine",
            "roles": ['Entity', 'Place', 'Adjudicator'],
        },
        "Justice:Pardon": {
            "template": "<arg1> received a pardon from <arg2> at <arg3> place",
            "roles": ['Defendant', 'Adjudicator', 'Place'],
        },
        "Justice:Release-Parole": {
            "template": "<arg1> was released by <arg2> from <arg3> place",
            "roles": ['Person', 'Agent', 'Place'],
        },
        "Justice:Sentence": {
            "template": "<arg1> was sentenced to punishment in <arg2> place, and the adjudication was judged by <arg3> adjudicator",
            "roles": ['Defendant', 'Place', 'Adjudicator'],
        },
        "Justice:Sue": {
            "template": "<arg1> was sued by <arg2> in <arg3> place, and the adjudication was judged by <arg4> adjudicator",
            "roles": ['Defendant', 'Plaintiff', 'Place', 'Adjudicator'],
        },
        "Justice:Trial-Hearing": {
            "template": "<arg1>, prosecuted by <arg2>, faced a trial in <arg3> place. The hearing was judged by <arg4> adjudicator",
            "roles": ['Defendant', 'Prosecutor', 'Place', 'Adjudicator'],
        },
        "Life:Be-Born": {
            "template": "<arg1> was born at <arg2> place",
            "roles": ['Person', 'Place'],
        },
        "Life:Die": {
            "template": "<arg1> led to <arg2> died by <arg3> at <arg4> place",
            "roles": ['Agent', 'Victim', 'Instrument', 'Place'],
        },
        "Life:Divorce": {
            "template": "<arg1> divorced at <arg2> place",
            "roles": ['Person', 'Place'],
        },
        "Life:Injure": {
            "template": "<arg1> led to <arg2> injured by <arg3> at <arg4> place",
            "roles": ['Agent', 'Victim', 'Instrument', 'Place'],
        },
        "Life:Marry": {
            "template": "<arg1> got married at <arg2> place",
            "roles": ['Person', 'Place'],
        },
        "Manufacture:Artifact": {
            "template": "<arg1> was built by <arg2> at <arg3> place",
            "roles": ['Artifact', 'Agent', 'Place'],
        },
        "Movement:Transport-Artifact": {
            "template": "<arg1> was sent to <arg2> from <arg3>, and <arg4> was responsible for the transport",
            "roles": ['Artifact', 'Destination', 'Origin', 'Agent'],
        },
        "Movement:Transport-Person": {
            "template": "<arg1> was moved to <arg2> from <arg3> by <arg4>, and <arg5> was responsible for the movement",
            "roles": ['Person', 'Destination', 'Origin', 'Instrument', 'Agent'],
        },
        "Personnel:Elect": {
            "template": "<arg1> was elected a position, and the election was voted by <arg2> at <arg3> place",
            "roles": ['Person', 'Agent', 'Place'],
        },
        "Personnel:End-Position": {
            "template": "<arg1> stopped working for <arg2> at <arg3> place",
            "roles": ['Person', 'Entity', 'Place'],
        },
        "Personnel:Nominate": {
            "template": "<arg1> was nominated by {ROLE_Entity} to do a job at <arg3> place",
            "roles": ['Person', 'Agent', 'Place'],
        },
        "Personnel:Start-Position": {
            "template": "<arg1> got new job and was hired by <arg2> at <arg3> place",
            "roles": ['Person', 'Entity', 'Place'],
        },
        "Transaction:Transaction": {
            "template": "<arg1> give some things to <arg2> at <arg3> place",
            "roles": ['Giver', 'Recipient', 'Place', 'Beneficiary'],
        },
        "Transaction:Transfer-Money": {
            "template": "<arg1> paid <arg2> at <arg3> place",
            "roles": ['Giver', 'Recipient', 'Place', 'Beneficiary'],
        },
        "Transaction:Transfer-Ownership": {
            "template": "<arg1> from <arg2> was transferred to <arg3> at <arg4> place",
            "roles": ['Thing', 'Giver', 'Recipient', 'Place', 'Beneficiary'],
        },
    },
    "geneva": {
        "Achieve": {
            "template": "<arg1> Agent <arg2> Goal <arg3> Means",
            "roles": ['Agent', 'Goal', 'Means'],
        },
        "Action": {
            "template": "<arg1> Act <arg2> Agent <arg3> Domain <arg4> Manner",
            "roles": ['Act', 'Agent', 'Domain', 'Manner'],
        },
        "Adducing": {
            "template": "<arg1> Medium <arg2> Role <arg3> Speaker <arg4> Specified Entity",
            "roles": ['Medium', 'Role', 'Speaker', 'Specified_entity'],
        },
        "Agree_or_refuse_to_act": {
            "template": "<arg1> Proposed Action <arg2> Speaker",
            "roles": ['Proposed_action', 'Speaker'],
        },
        "Arranging": {
            "template": "<arg1> Agent <arg2> Configuration <arg3> Location <arg4> Theme",
            "roles": ['Agent', 'Configuration', 'Location', 'Theme'],
        },
        "Arrest": {
            "template": "<arg1> Authorities <arg2> Charges <arg3> Offense <arg4> Suspect",
            "roles": ['Authorities', 'Charges', 'Offense', 'Suspect'],
        },
        "Arriving": {
            "template": "<arg1> Goal <arg2> Means <arg3> Path <arg4> Place <arg5> Purpose <arg6> Source <arg7> Theme",
            "roles": ['Goal', 'Means', 'Path', 'Place', 'Purpose', 'Source', 'Theme'],
        },
        "Assistance": {
            "template": "<arg1> Benefited Party <arg2> Focal Entity <arg3> Goal <arg4> Helper <arg5> Means",
            "roles": ['Benefited_party', 'Focal_entity', 'Goal', 'Helper', 'Means'],
        },
        "Attack": {
            "template": "<arg1> Assailant <arg2> Means <arg3> Victim <arg4> Weapon",
            "roles": ['Assailant', 'Means', 'Victim', 'Weapon'],
        },
        "Bearing_arms": {
            "template": "<arg1> Protagonist <arg2> Weapon",
            "roles": ['Protagonist', 'Weapon'],
        },
        "Becoming": {
            "template": "<arg1> Entity <arg2> Final Category <arg3> Final Quality",
            "roles": ['Entity', 'Final_category', 'Final_quality'],
        },
        "Becoming_a_member": {
            "template": "<arg1> Group <arg2> New Member",
            "roles": ['Group', 'New_member'],
        },
        "Bodily_harm": {
            "template": "<arg1> Agent <arg2> Body Part <arg3> Cause <arg4> Instrument <arg5> Victim",
            "roles": ['Agent', 'Body_part', 'Cause', 'Instrument', 'Victim'],
        },
        "Bringing": {
            "template": "<arg1> Agent <arg2> Area <arg3> Carrier <arg4> Goal <arg5> Source <arg6> Theme",
            "roles": ['Agent', 'Area', 'Carrier', 'Goal', 'Source', 'Theme'],
        },
        "Building": {
            "template": "<arg1> Agent <arg2> Components <arg3> Created Entity <arg4> Place",
            "roles": ['Agent', 'Components', 'Created_entity', 'Place'],
        },
        "Catastrophe": {
            "template": "<arg1> Cause <arg2> Patient <arg3> Place <arg4> Undesirable Event",
            "roles": ['Cause', 'Patient', 'Place', 'Undesirable_event'],
        },
        "Causation": {
            "template": "<arg1> Actor <arg2> Affected <arg3> Cause <arg4> Effect <arg5> Means",
            "roles": ['Actor', 'Affected', 'Cause', 'Effect', 'Means'],
        },
        "Cause_change_of_position_on_a_scale": {
            "template": "<arg1> Agent <arg2> Attribute <arg3> Cause <arg4> Difference <arg5> Item <arg6> Value 1 <arg7> Value 2",
            "roles": ['Agent', 'Attribute', 'Cause', 'Difference', 'Item', 'Value_1', 'Value_2'],
        },
        "Cause_to_amalgamate": {
            "template": "<arg1> Agent <arg2> Part 1 <arg3> Part 2 <arg4> Parts <arg5> Whole",
            "roles": ['Agent', 'Part_1', 'Part_2', 'Parts', 'Whole'],
        },
        "Cause_to_make_progress": {
            "template": "<arg1> Agent <arg2> Project",
            "roles": ['Agent', 'Project'],
        },
        "Change": {
            "template": "<arg1> Agent <arg2> Attribute <arg3> Cause <arg4> Entity <arg5> Final Category <arg6> Final Value <arg7> Initial Category",
            "roles": ['Agent', 'Attribute', 'Cause', 'Entity', 'Final_category', 'Final_value', 'Initial_category'],
        },
        "Change_of_leadership": {
            "template": "<arg1> Body <arg2> New Leader <arg3> Old Leader <arg4> Old Order <arg5> Role <arg6> Selector",
            "roles": ['Body', 'New_leader', 'Old_leader', 'Old_order', 'Role', 'Selector'],
        },
        "Check": {
            "template": "<arg1> Inspector <arg2> Means <arg3> Unconfirmed Content",
            "roles": ['Inspector', 'Means', 'Unconfirmed_content'],
        },
        "Choosing": {
            "template": "<arg1> Chosen <arg2> Cognizer <arg3> Possibilities",
            "roles": ['Chosen', 'Cognizer', 'Possibilities'],
        },
        "Collaboration": {
            "template": "<arg1> Partners <arg2> Undertaking",
            "roles": ['Partners', 'Undertaking'],
        },
        "Come_together": {
            "template": "<arg1> Configuration <arg2> Individuals <arg3> Place",
            "roles": ['Configuration', 'Individuals', 'Place'],
        },
        "Coming_to_be": {
            "template": "<arg1> Components <arg2> Entity <arg3> Place <arg4> Time",
            "roles": ['Components', 'Entity', 'Place', 'Time'],
        },
        "Coming_to_believe": {
            "template": "<arg1> Cognizer <arg2> Content <arg3> Means",
            "roles": ['Cognizer', 'Content', 'Means'],
        },
        "Commerce_buy": {
            "template": "<arg1> Buyer <arg2> Goods <arg3> Seller",
            "roles": ['Buyer', 'Goods', 'Seller'],
        },
        "Commerce_pay": {
            "template": "<arg1> Buyer <arg2> Goods <arg3> Money <arg4> Seller",
            "roles": ['Buyer', 'Goods', 'Money', 'Seller'],
        },
        "Commerce_sell": {
            "template": "<arg1> Buyer <arg2> Goods <arg3> Money <arg4> Seller",
            "roles": ['Buyer', 'Goods', 'Money', 'Seller'],
        },
        "Commitment": {
            "template": "<arg1> Addressee <arg2> Message <arg3> Speaker",
            "roles": ['Addressee', 'Message', 'Speaker'],
        },
        "Committing_crime": {
            "template": "<arg1> Crime <arg2> Instrument <arg3> Perpetrator",
            "roles": ['Crime', 'Instrument', 'Perpetrator'],
        },
        "Communication": {
            "template": "<arg1> Addressee <arg2> Interlocutors <arg3> Message <arg4> Speaker <arg5> Topic <arg6> Trigger",
            "roles": ['Addressee', 'Interlocutors', 'Message', 'Speaker', 'Topic', 'Trigger'],
        },
        "Competition": {
            "template": "<arg1> Competition <arg2> Participants <arg3> Venue",
            "roles": ['Competition', 'Participants', 'Venue'],
        },
        "Confronting_problem": {
            "template": "<arg1> Activity <arg2> Experiencer",
            "roles": ['Activity', 'Experiencer'],
        },
        "Connect": {
            "template": "<arg1> Figures <arg2> Ground",
            "roles": ['Figures', 'Ground'],
        },
        "Conquering": {
            "template": "<arg1> Conqueror <arg2> Means <arg3> Theme",
            "roles": ['Conqueror', 'Means', 'Theme'],
        },
        "Containing": {
            "template": "<arg1> Container <arg2> Contents",
            "roles": ['Container', 'Contents'],
        },
        "Control": {
            "template": "<arg1> Controlling Variable <arg2> Dependent Variable",
            "roles": ['Controlling_variable', 'Dependent_variable'],
        },
        "Convincing": {
            "template": "<arg1> Addressee <arg2> Content <arg3> Speaker <arg4> Topic",
            "roles": ['Addressee', 'Content', 'Speaker', 'Topic'],
        },
        "Cost": {
            "template": "<arg1> Asset <arg2> Goods <arg3> Intended Event <arg4> Payer <arg5> Rate",
            "roles": ['Asset', 'Goods', 'Intended_event', 'Payer', 'Rate'],
        },
        "Create_artwork": {
            "template": "<arg1> Activity <arg2> Culture",
            "roles": ['Activity', 'Culture'],
        },
        "Creating": {
            "template": "<arg1> Cause <arg2> Created Entity <arg3> Creator",
            "roles": ['Cause', 'Created_entity', 'Creator'],
        },
        "Criminal_investigation": {
            "template": "<arg1> Incident <arg2> Investigator <arg3> Suspect",
            "roles": ['Incident', 'Investigator', 'Suspect'],
        },
        "Cure": {
            "template": "<arg1> Affliction <arg2> Medication <arg3> Patient <arg4> Treatment",
            "roles": ['Affliction', 'Medication', 'Patient', 'Treatment'],
        },
        "Damaging": {
            "template": "<arg1> Agent <arg2> Cause <arg3> Patient",
            "roles": ['Agent', 'Cause', 'Patient'],
        },
        "Death": {
            "template": "<arg1> Deceased",
            "roles": ['Deceased'],
        },
        "Deciding": {
            "template": "<arg1> Cognizer <arg2> Decision",
            "roles": ['Cognizer', 'Decision'],
        },
        "Defending": {
            "template": "<arg1> Assailant <arg2> Defender <arg3> Instrument <arg4> Victim",
            "roles": ['Assailant', 'Defender', 'Instrument', 'Victim'],
        },
        "Departing": {
            "template": "<arg1> Goal <arg2> Path <arg3> Source <arg4> Traveller",
            "roles": ['Goal', 'Path', 'Source', 'Traveller'],
        },
        "Destroying": {
            "template": "<arg1> Cause <arg2> Destroyer <arg3> Means <arg4> Patient",
            "roles": ['Cause', 'Destroyer', 'Means', 'Patient'],
        },
        "Dispersal": {
            "template": "<arg1> Agent <arg2> Cause <arg3> Goal Area <arg4> Individuals",
            "roles": ['Agent', 'Cause', 'Goal_area', 'Individuals'],
        },
        "Earnings_and_losses": {
            "template": "<arg1> Earner <arg2> Earnings <arg3> Goods",
            "roles": ['Earner', 'Earnings', 'Goods'],
        },
        "Education_teaching": {
            "template": "<arg1> Course <arg2> Fact <arg3> Role <arg4> Skill <arg5> Student <arg6> Subject <arg7> Teacher",
            "roles": ['Course', 'Fact', 'Role', 'Skill', 'Student', 'Subject', 'Teacher'],
        },
        "Emergency": {
            "template": "<arg1> Place <arg2> Undesirable Event",
            "roles": ['Place', 'Undesirable_event'],
        },
        "Employment": {
            "template": "<arg1> Employee <arg2> Employer <arg3> Field <arg4> Place Of Employment <arg5> Position <arg6> Task <arg7> Type",
            "roles": ['Employee', 'Employer', 'Field', 'Place_of_employment', 'Position', 'Task', 'Type'],
        },
        "Emptying": {
            "template": "<arg1> Agent <arg2> Cause <arg3> Source <arg4> Theme",
            "roles": ['Agent', 'Cause', 'Source', 'Theme'],
        },
        "Exchange": {
            "template": "<arg1> Exchanger 1 <arg2> Exchanger 2 <arg3> Exchangers <arg4> Theme 1 <arg5> Theme 2 <arg6> Themes",
            "roles": ['Exchanger_1', 'Exchanger_2', 'Exchangers', 'Theme_1', 'Theme_2', 'Themes'],
        },
        "Expansion": {
            "template": "<arg1> Agent <arg2> Dimension <arg3> Initial Size <arg4> Item <arg5> Result Size",
            "roles": ['Agent', 'Dimension', 'Initial_size', 'Item', 'Result_size'],
        },
        "Filling": {
            "template": "<arg1> Agent <arg2> Cause <arg3> Goal <arg4> Theme",
            "roles": ['Agent', 'Cause', 'Goal', 'Theme'],
        },
        "GetReady": {
            "template": "<arg1> Activity <arg2> Protagonist",
            "roles": ['Activity', 'Protagonist'],
        },
        "Getting": {
            "template": "<arg1> Means <arg2> Recipient <arg3> Source <arg4> Theme",
            "roles": ['Means', 'Recipient', 'Source', 'Theme'],
        },
        "Giving": {
            "template": "<arg1> Offerer <arg2> Recipient <arg3> Theme",
            "roles": ['Offerer', 'Recipient', 'Theme'],
        },
        "Hindering": {
            "template": "<arg1> Action <arg2> Hindrance <arg3> Protagonist",
            "roles": ['Action', 'Hindrance', 'Protagonist'],
        },
        "Hold": {
            "template": "<arg1> Entity <arg2> Manipulator",
            "roles": ['Entity', 'Manipulator'],
        },
        "Hostile_encounter": {
            "template": "<arg1> Instrument <arg2> Issue <arg3> Purpose <arg4> Side 1 <arg5> Side 2 <arg6> Sides",
            "roles": ['Instrument', 'Issue', 'Purpose', 'Side_1', 'Side_2', 'Sides'],
        },
        "Influence": {
            "template": "<arg1> Action <arg2> Agent <arg3> Behavior <arg4> Cognizer <arg5> Product <arg6> Situation",
            "roles": ['Action', 'Agent', 'Behavior', 'Cognizer', 'Product', 'Situation'],
        },
        "Ingestion": {
            "template": "<arg1> Ingestibles <arg2> Ingestor",
            "roles": ['Ingestibles', 'Ingestor'],
        },
        "Judgment_communication": {
            "template": "<arg1> Addressee <arg2> Communicator <arg3> Evaluee <arg4> Medium <arg5> Reason <arg6> Topic",
            "roles": ['Addressee', 'Communicator', 'Evaluee', 'Medium', 'Reason', 'Topic'],
        },
        "Killing": {
            "template": "<arg1> Cause <arg2> Instrument <arg3> Killer <arg4> Means <arg5> Victim",
            "roles": ['Cause', 'Instrument', 'Killer', 'Means', 'Victim'],
        },
        "Know": {
            "template": "<arg1> Cognizer <arg2> Evidence <arg3> Instrument <arg4> Means <arg5> Phenomenon <arg6> Topic",
            "roles": ['Cognizer', 'Evidence', 'Instrument', 'Means', 'Phenomenon', 'Topic'],
        },
        "Labeling": {
            "template": "<arg1> Entity <arg2> Label <arg3> Speaker",
            "roles": ['Entity', 'Label', 'Speaker'],
        },
        "Legality": {
            "template": "<arg1> Action <arg2> Object",
            "roles": ['Action', 'Object'],
        },
        "Manufacturing": {
            "template": "<arg1> Factory <arg2> Instrument <arg3> Producer <arg4> Product <arg5> Resource",
            "roles": ['Factory', 'Instrument', 'Producer', 'Product', 'Resource'],
        },
        "Motion": {
            "template": "<arg1> Agent <arg2> Area <arg3> Cause <arg4> Distance <arg5> Goal <arg6> Means <arg7> Path <arg8> Source <arg9> Speed <arg10> Theme",
            "roles": ['Agent', 'Area', 'Cause', 'Distance', 'Goal', 'Means', 'Path', 'Source', 'Speed', 'Theme'],
        },
        "Motion_directional": {
            "template": "<arg1> Area <arg2> Direction <arg3> Goal <arg4> Path <arg5> Source <arg6> Theme",
            "roles": ['Area', 'Direction', 'Goal', 'Path', 'Source', 'Theme'],
        },
        "Openness": {
            "template": "<arg1> Barrier <arg2> Theme <arg3> Useful Location",
            "roles": ['Barrier', 'Theme', 'Useful_location'],
        },
        "Participation": {
            "template": "<arg1> Event <arg2> Participants",
            "roles": ['Event', 'Participants'],
        },
        "Perception_active": {
            "template": "<arg1> Direction <arg2> Perceiver Agentive <arg3> Phenomenon",
            "roles": ['Direction', 'Perceiver_agentive', 'Phenomenon'],
        },
        "Placing": {
            "template": "<arg1> Agent <arg2> Cause <arg3> Location <arg4> Theme",
            "roles": ['Agent', 'Cause', 'Location', 'Theme'],
        },
        "Practice": {
            "template": "<arg1> Action <arg2> Agent <arg3> Occasion",
            "roles": ['Action', 'Agent', 'Occasion'],
        },
        "Presence": {
            "template": "<arg1> Circumstances <arg2> Duration <arg3> Entity <arg4> Inherent Purpose <arg5> Place <arg6> Time",
            "roles": ['Circumstances', 'Duration', 'Entity', 'Inherent_purpose', 'Place', 'Time'],
        },
        "Preventing_or_letting": {
            "template": "<arg1> Agent <arg2> Event <arg3> Means <arg4> Potential Hindrance",
            "roles": ['Agent', 'Event', 'Means', 'Potential_hindrance'],
        },
        "Process_end": {
            "template": "<arg1> Agent <arg2> Cause <arg3> Final Subevent <arg4> Process <arg5> State <arg6> Time",
            "roles": ['Agent', 'Cause', 'Final_subevent', 'Process', 'State', 'Time'],
        },
        "Process_start": {
            "template": "<arg1> Agent <arg2> Event <arg3> Initial Subevent <arg4> Time",
            "roles": ['Agent', 'Event', 'Initial_subevent', 'Time'],
        },
        "Protest": {
            "template": "<arg1> Addressee <arg2> Arguer <arg3> Content",
            "roles": ['Addressee', 'Arguer', 'Content'],
        },
        "Quarreling": {
            "template": "<arg1> Arguer2 <arg2> Arguers <arg3> Issue",
            "roles": ['Arguer2', 'Arguers', 'Issue'],
        },
        "Ratification": {
            "template": "<arg1> Proposal <arg2> Ratifier",
            "roles": ['Proposal', 'Ratifier'],
        },
        "Receiving": {
            "template": "<arg1> Donor <arg2> Recipient <arg3> Theme",
            "roles": ['Donor', 'Recipient', 'Theme'],
        },
        "Recovering": {
            "template": "<arg1> Agent <arg2> Entity <arg3> Means",
            "roles": ['Agent', 'Entity', 'Means'],
        },
        "Removing": {
            "template": "<arg1> Agent <arg2> Cause <arg3> Goal <arg4> Source <arg5> Theme",
            "roles": ['Agent', 'Cause', 'Goal', 'Source', 'Theme'],
        },
        "Request": {
            "template": "<arg1> Addressee <arg2> Medium <arg3> Message <arg4> Speaker",
            "roles": ['Addressee', 'Medium', 'Message', 'Speaker'],
        },
        "Research": {
            "template": "<arg1> Field <arg2> Researcher <arg3> Topic",
            "roles": ['Field', 'Researcher', 'Topic'],
        },
        "Resolve_problem": {
            "template": "<arg1> Agent <arg2> Cause <arg3> Means <arg4> Problem",
            "roles": ['Agent', 'Cause', 'Means', 'Problem'],
        },
        "Response": {
            "template": "<arg1> Agent <arg2> Responding Entity <arg3> Response <arg4> Trigger",
            "roles": ['Agent', 'Responding_entity', 'Response', 'Trigger'],
        },
        "Reveal_secret": {
            "template": "<arg1> Addressee <arg2> Information <arg3> Speaker <arg4> Topic",
            "roles": ['Addressee', 'Information', 'Speaker', 'Topic'],
        },
        "Revenge": {
            "template": "<arg1> Avenger <arg2> Injured Party <arg3> Injury <arg4> Offender <arg5> Punishment",
            "roles": ['Avenger', 'Injured_party', 'Injury', 'Offender', 'Punishment'],
        },
        "Rite": {
            "template": "<arg1> Member <arg2> Object <arg3> Type",
            "roles": ['Member', 'Object', 'Type'],
        },
        "Scrutiny": {
            "template": "<arg1> Cognizer <arg2> Ground <arg3> Instrument <arg4> Phenomenon <arg5> Unwanted Entity",
            "roles": ['Cognizer', 'Ground', 'Instrument', 'Phenomenon', 'Unwanted_entity'],
        },
        "Self_motion": {
            "template": "<arg1> Direction <arg2> Distance <arg3> Goal <arg4> Path <arg5> Self Mover <arg6> Source",
            "roles": ['Direction', 'Distance', 'Goal', 'Path', 'Self_mover', 'Source'],
        },
        "Sending": {
            "template": "<arg1> Goal <arg2> Means <arg3> Recipient <arg4> Sender <arg5> Source <arg6> Theme <arg7> Transferors <arg8> Vehicle",
            "roles": ['Goal', 'Means', 'Recipient', 'Sender', 'Source', 'Theme', 'Transferors', 'Vehicle'],
        },
        "Sign_agreement": {
            "template": "<arg1> Agreement <arg2> Signatory",
            "roles": ['Agreement', 'Signatory'],
        },
        "Social_event": {
            "template": "<arg1> Attendees <arg2> Beneficiary <arg3> Host <arg4> Occasion <arg5> Social Event",
            "roles": ['Attendees', 'Beneficiary', 'Host', 'Occasion', 'Social_event'],
        },
        "Statement": {
            "template": "<arg1> Addressee <arg2> Medium <arg3> Message <arg4> Speaker",
            "roles": ['Addressee', 'Medium', 'Message', 'Speaker'],
        },
        "Supply": {
            "template": "<arg1> Imposed Purpose <arg2> Recipient <arg3> Supplier <arg4> Theme",
            "roles": ['Imposed_purpose', 'Recipient', 'Supplier', 'Theme'],
        },
        "Supporting": {
            "template": "<arg1> Supported <arg2> Supporter",
            "roles": ['Supported', 'Supporter'],
        },
        "Telling": {
            "template": "<arg1> Addressee <arg2> Message <arg3> Speaker",
            "roles": ['Addressee', 'Message', 'Speaker'],
        },
        "Terrorism": {
            "template": "<arg1> Act <arg2> Instrument <arg3> Terrorist <arg4> Victim",
            "roles": ['Act', 'Instrument', 'Terrorist', 'Victim'],
        },
        "Testing": {
            "template": "<arg1> Circumstances <arg2> Means <arg3> Product <arg4> Result <arg5> Tested Property <arg6> Tester",
            "roles": ['Circumstances', 'Means', 'Product', 'Result', 'Tested_property', 'Tester'],
        },
        "Theft": {
            "template": "<arg1> Goods <arg2> Instrument <arg3> Means <arg4> Perpetrator <arg5> Source <arg6> Victim",
            "roles": ['Goods', 'Instrument', 'Means', 'Perpetrator', 'Source', 'Victim'],
        },
        "Traveling": {
            "template": "<arg1> Area <arg2> Distance <arg3> Entity <arg4> Goal <arg5> Means <arg6> Path <arg7> Purpose <arg8> Traveler",
            "roles": ['Area', 'Distance', 'Entity', 'Goal', 'Means', 'Path', 'Purpose', 'Traveler'],
        },
        "Using": {
            "template": "<arg1> Agent <arg2> Instrument <arg3> Means <arg4> Purpose",
            "roles": ['Agent', 'Instrument', 'Means', 'Purpose'],
        },
        "Wearing": {
            "template": "<arg1> Body Location <arg2> Clothing <arg3> Wearer",
            "roles": ['Body_location', 'Clothing', 'Wearer'],
        },
        "Writing": {
            "template": "<arg1> Addressee <arg2> Author <arg3> Instrument <arg4> Text",
            "roles": ['Addressee', 'Author', 'Instrument', 'Text'],
        },
    },
    "m2e2": {
        "Conflict:Attack": {
            "template": "<arg1> attacked <arg2> using <arg3> instrument at <arg4> place",
            "roles": ["Attacker", "Target", "Instrument", "Place"],
        },
        "Conflict:Demonstrate": {
            "template": "<arg1> demonstrated with <arg2> at <arg3> place",
            "roles": ["Entity", "Police" ,"Place"],
        },
        "Contact:Meet": {
            "template": "<arg1> met with <arg2> in <arg3> place",
            "roles": ["Entity", "Entity", "Place"],
        },
        "Contact:Phone-Write": {
            "template": "<arg1> communicated remotely with <arg2> at <arg3> place",
            "roles": ["Entity", "Entity", "Place"],
        },
        "Justice:Arrest-Jail": {
            "template": "<arg1> arrested <arg2> in <arg3> place",
            "roles": ["Agent", "Person", "Place"],
        },
        "Life:Die": {
            "template": "<arg1> killed <arg2> with <arg3> instrument in <arg4> place",
            "roles": ["Agent", "Victim", "Instrument", "Place"],
        },
        "Movement:Transport": {
            "template": "<arg1> transported <arg2> in <arg3> vehicle from <arg4> place to <arg5> place",
            "roles": ["Agent", "Artifact", "Vehicle", "Origin", "Destination"],
        },
        "Transaction:Transfer-Money": {
            "template": "<arg1> gave money to <arg2>",
            "roles": ["Giver", "Recipient", "Beneficiary", "Place"],
        },  
    }, 
    "rams": {
        "artifactexistence.damagedestroy.damage": {
            "template": "<arg1> damaged <arg2> using <arg3> instrument in <arg4> place",
            "roles": ['damager', 'artifact', 'instrument', 'place']
        },
        "artifactexistence.damagedestroy.destroy": {
            "template": "<arg1> destroyed <arg2> using <arg3> instrument in <arg4> place",
            "roles": ['destroyer', 'artifact', 'instrument', 'place']
        },
        "artifactexistence.damagedestroy.n/a": {
            "template": "<arg1> damaged or destroyed <arg2> using <arg3> instrument in <arg4> place",
            "roles": ['damagerdestroyer', 'artifact', 'instrument', 'place']
        },
        "conflict.attack.airstrikemissilestrike": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.biologicalchemicalpoisonattack": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.bombing": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.firearmattack": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.hanging": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.invade": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.n/a": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.selfdirectedbattle": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.setfire": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.stabbing": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.attack.stealrobhijack": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place, in order to take <arg5>",
            "roles": ['attacker', 'target', 'instrument', 'place', 'artifact']
        },
        "conflict.attack.strangling": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['attacker', 'target', 'instrument', 'place']
        },
        "conflict.demonstrate.marchprotestpoliticalgathering": {
            "template": "<arg1> was in a demonstration or protest at <arg2> place",
            "roles": ['demonstrator', 'place']
        },
        "conflict.demonstrate.n/a": {
            "template": "<arg1> was in a demonstration at <arg2> place",
            "roles": ['demonstrator', 'place']
        },
        "conflict.yield.n/a": {
            "template": "<arg1> yielded to <arg2> at <arg3> place",
            "roles": ['yielder', 'recipient', 'place']
        },
        "conflict.yield.retreat": {
            "template": "<arg1> retreated from <arg2> place to <arg3> place",
            "roles": ['retreater', 'origin', 'destination']
        },
        "conflict.yield.surrender": {
            "template": "<arg1> surrendered to <arg2> at <arg3> place",
            "roles": ['surrenderer', 'recipient', 'place']
        },
        "contact.collaborate.correspondence": {
            "template": "<arg1> communicated remotely with <arg2> at <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "contact.collaborate.meet": {
            "template": "<arg1> met face-to-face with <arg2> at <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "contact.collaborate.n/a": {
            "template": "<arg1> communicated with <arg2> at <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "contact.commandorder.broadcast": {
            "template": "<arg1> communicated to <arg2> about <arg4> topic at <arg3> place (one-way communication)",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.commandorder.correspondence": {
            "template": "<arg1> communicated remotely with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.commandorder.meet": {
            "template": "<arg1> met face-to-face with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.commandorder.n/a": {
            "template": "<arg1> communicated with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.commitmentpromiseexpressintent.broadcast": {
            "template": "<arg1> communicated to <arg2> about <arg4> topic at <arg3> place (one-way communication)",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.commitmentpromiseexpressintent.correspondence": {
            "template": "<arg1> communicated remotely with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.commitmentpromiseexpressintent.meet": {
            "template": "<arg1> met face-to-face with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.commitmentpromiseexpressintent.n/a": {
            "template": "<arg1> communicated with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.discussion.correspondence": {
            "template": "<arg1> communicated remotely with <arg2> at <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "contact.discussion.meet": {
            "template": "<arg1> met face-to-face with <arg2> at <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "contact.discussion.n/a": {
            "template": "<arg1> communicated with <arg2> at <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "contact.funeralvigil.meet": {
            "template": "<arg1> met face-to-face with <arg2> during a funeral or vigil for <arg3> at <arg4> place",
            "roles": ['participant', 'participant', 'deceased', 'place']
        },
        "contact.funeralvigil.n/a": {
            "template": "<arg1> communicated with <arg2> during a funeral or vigil for <arg3> at <arg4> place",
            "roles": ['participant', 'participant', 'deceased', 'place']
        },
        "contact.mediastatement.broadcast": {
            "template": "<arg1> communicated to <arg2> at <arg3> place (one-way communication)",
            "roles": ['communicator', 'recipient', 'place']
        },
        "contact.mediastatement.n/a": {
            "template": "<arg1> communicated with <arg2> at <arg3> place",
            "roles": ['communicator', 'recipient', 'place']
        },
        "contact.negotiate.correspondence": {
            "template": "<arg1> communicated remotely with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['participant', 'participant', 'place', 'topic']
        },
        "contact.negotiate.meet": {
            "template": "<arg1> met face-to-face with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['participant', 'participant', 'place', 'topic']
        },
        "contact.negotiate.n/a": {
            "template": "<arg1> communicated with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['participant', 'participant', 'place', 'topic']
        },
        "contact.prevarication.broadcast": {
            "template": "<arg1> communicated to <arg2> about <arg4> topic at <arg3> place (one-way communication)",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.prevarication.correspondence": {
            "template": "<arg1> communicated remotely with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.prevarication.meet": {
            "template": "<arg1> met face-to-face with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.prevarication.n/a": {
            "template": "<arg1> communicated with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.publicstatementinperson.broadcast": {
            "template": "<arg1> communicated to <arg2> at <arg3> place (one-way communication)",
            "roles": ['communicator', 'recipient', 'place']
        },
        "contact.publicstatementinperson.n/a": {
            "template": "<arg1> communicated with <arg2> at <arg3> place",
            "roles": ['communicator', 'recipient', 'place']
        },
        "contact.requestadvise.broadcast": {
            "template": "<arg1> communicated to <arg2> about <arg4> topic at <arg3> place (one-way communication)",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.requestadvise.correspondence": {
            "template": "<arg1> communicated remotely with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.requestadvise.meet": {
            "template": "<arg1> met face-to-face with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.requestadvise.n/a": {
            "template": "<arg1> communicated with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.threatencoerce.broadcast": {
            "template": "<arg1> communicated to <arg2> about <arg4> topic at <arg3> place (one-way communication)",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.threatencoerce.correspondence": {
            "template": "<arg1> communicated remotely with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.threatencoerce.meet": {
            "template": "<arg1> met face-to-face with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "contact.threatencoerce.n/a": {
            "template": "<arg1> communicated with <arg2> about <arg4> topic at <arg3> place",
            "roles": ['communicator', 'recipient', 'place', 'topic']
        },
        "disaster.accidentcrash.accidentcrash": {
            "template": "<arg1> person in <arg2> vehicle crashed into <arg3> at <arg4> place",
            "roles": ['driverpassenger', 'vehicle', 'crashobject', 'place']
        },
        "disaster.fireexplosion.fireexplosion": {
            "template": "<arg1> caught fire or exploded from <arg2> instrument at <arg3> place",
            "roles": ['fireexplosionobject', 'instrument', 'place']
        },
        "government.agreements.acceptagreementcontractceasefire": {
            "template": "<arg1> and <arg2> signed an agreement in <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "government.agreements.n/a": {
            "template": "<arg1> and <arg2> signed an agreement in <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "government.agreements.rejectnullifyagreementcontractceasefire": {
            "template": "<arg1> rejected or nullified an agreement with <arg2> in <arg3> place",
            "roles": ['rejecternullifier', 'otherparticipant', 'place']
        },
        "government.agreements.violateagreement": {
            "template": "<arg1> violated an agreement with <arg2> in <arg3> place",
            "roles": ['violator', 'otherparticipant', 'place']
        },
        "government.formation.mergegpe": {
            "template": "<arg1> merged with <arg2> at <arg3> place",
            "roles": ['participant', 'participant', 'place']
        },
        "government.formation.n/a": {
            "template": "<arg1> was formed by <arg2> in <arg3> place",
            "roles": ['gpe', 'founder', 'place']
        },
        "government.formation.startgpe": {
            "template": "<arg1> was started by <arg2> in <arg3> place",
            "roles": ['gpe', 'founder', 'place']
        },
        "government.legislate.legislate": {
            "template": "<arg1> legislature enacted <arg2> law in <arg3> place",
            "roles": ['governmentbody', 'law', 'place']
        },
        "government.spy.spy": {
            "template": "<arg1> spied on <arg2> to the benefit of <arg3> in <arg4> place",
            "roles": ['spy', 'observedentity', 'beneficiary', 'place']
        },
        "government.vote.castvote": {
            "template": "<arg1> voted for <arg2> on <arg3> ballot with <arg4> results in <arg5> place",
            "roles": ['voter', 'candidate', 'ballot', 'result', 'place']
        },
        "government.vote.n/a": {
            "template": "<arg1> voted for <arg2> on <arg3> ballot with <arg4> results in <arg5> place",
            "roles": ['voter', 'candidate', 'ballot', 'result', 'place']
        },
        "government.vote.violationspreventvote": {
            "template": "<arg1> prevented <arg2> from voting for <arg3> on <arg4> ballot in <arg5> place",
            "roles": ['preventer', 'voter', 'candidate', 'ballot', 'place']
        },
        "inspection.sensoryobserve.inspectpeopleorganization": {
            "template": "<arg1> inspected <arg2> in <arg3> place",
            "roles": ['inspector', 'inspectedentity', 'place']
        },
        "inspection.sensoryobserve.monitorelection": {
            "template": "<arg1> monitored <arg2> taking part in an election in <arg3> place",
            "roles": ['monitor', 'monitoredentity', 'place']
        },
        "inspection.sensoryobserve.n/a": {
            "template": "<arg1> observed <arg2> in <arg3> place",
            "roles": ['observer', 'observedentity', 'place']
        },
        "inspection.sensoryobserve.physicalinvestigateinspect": {
            "template": "<arg1> inspected <arg2> in <arg3> place",
            "roles": ['inspector', 'inspectedentity', 'place']
        },
        "justice.arrestjaildetain.arrestjaildetain": {
            "template": "<arg1> arrested or jailed <arg2> for <arg3> crime at <arg4> place",
            "roles": ['jailer', 'detainee', 'crime', 'place']
        },
        "justice.initiatejudicialprocess.chargeindict": {
            "template": "<arg1> charged or indicted <arg2> before <arg3> court or judge for <arg4> crime in <arg5> place",
            "roles": ['prosecutor', 'defendant', 'judgecourt', 'crime', 'place']
        },
        "justice.initiatejudicialprocess.n/a": {
            "template": "<arg1> initiated judicial process pertaining to <arg2> before <arg3> court or judge for <arg4> crime in <arg5> place",
            "roles": ['prosecutor', 'defendant', 'judgecourt', 'crime', 'place']
        },
        "justice.initiatejudicialprocess.trialhearing": {
            "template": "<arg1> tried <arg2> before <arg3> court or judge for <arg4> crime in <arg5> place",
            "roles": ['prosecutor', 'defendant', 'judgecourt', 'crime', 'place']
        },
        "justice.investigate.investigatecrime": {
            "template": "<arg1> investigated <arg2> for <arg3> crime in <arg4> place",
            "roles": ['investigator', 'defendant', 'crime', 'place']
        },
        "justice.investigate.n/a": {
            "template": "<arg1> investigated <arg2> in <arg3> place",
            "roles": ['investigator', 'defendant', 'place']
        },
        "justice.judicialconsequences.convict": {
            "template": "<arg1> court or judge convicted <arg2> of <arg3> crime in <arg4> place",
            "roles": ['judgecourt', 'defendant', 'crime', 'place']
        },
        "justice.judicialconsequences.execute": {
            "template": "<arg1> executed <arg2> for <arg3> crime in <arg4> place",
            "roles": ['executioner', 'defendant', 'crime', 'place']
        },
        "justice.judicialconsequences.extradite": {
            "template": "<arg1> extradited <arg2> for <arg3> crime from <arg4> place to <arg5> place",
            "roles": ['extraditer', 'defendant', 'crime', 'origin', 'destination']
        },
        "justice.judicialconsequences.n/a": {
            "template": "<arg1> court or judge decided consequences of <arg3> crime, committed by <arg2>, in <arg4> place",
            "roles": ['judgecourt', 'defendant', 'crime', 'place']
        },
        "life.die.deathcausedbyviolentevents": {
            "template": "<arg1> killed <arg2> using <arg3> instrument or <arg5> medical issue at <arg4> place",
            "roles": ['killer', 'victim', 'instrument', 'place', 'medicalissue']
        },
        "life.die.n/a": {
            "template": "<arg1> died at <arg2> place from <arg4> medical issue, killed by <arg3> killer",
            "roles": ['victim', 'place', 'killer', 'medicalissue']
        },
        "life.die.nonviolentdeath": {
            "template": "<arg1> died at <arg2> place from <arg4> medical issue, killed by <arg3> killer",
            "roles": ['victim', 'place', 'killer', 'medicalissue']
        },
        "life.injure.illnessdegradationhungerthirst": {
            "template": "<arg1> has extreme hunger or thirst from <arg4> medical issue imposed by <arg3> injurer at <arg2> place",
            "roles": ['victim', 'place', 'injurer', 'medicalissue']
        },
        "life.injure.illnessdegradationphysical": {
            "template": "<arg1> person has some physical degradation from <arg4> medical issue imposed by <arg3> injurer at <arg2> place",
            "roles": ['victim', 'place', 'injurer', 'medicalissue']
        },
        "life.injure.injurycausedbyviolentevents": {
            "template": "<arg1> injured <arg2> using <arg3> instrument or <arg5> medical issue at <arg4> place",
            "roles": ['injurer', 'victim', 'instrument', 'place', 'medicalissue']
        },
        "life.injure.n/a": {
            "template": "<arg1> was injured by <arg2> with <arg4> medical issue at <arg3> place",
            "roles": ['victim', 'injurer', 'place', 'medicalissue']
        },
        "manufacture.artifact.build": {
            "template": "<arg1> manufactured or created or produced <arg2> using <arg3> at <arg4> place",
            "roles": ['manufacturer', 'artifact', 'instrument', 'place']
        },
        "manufacture.artifact.createintellectualproperty": {
            "template": "<arg1> manufactured or created or produced <arg2> using <arg3> at <arg4> place",
            "roles": ['manufacturer', 'artifact', 'instrument', 'place']
        },
        "manufacture.artifact.createmanufacture": {
            "template": "<arg1> manufactured or created or produced <arg2> using <arg3> at <arg4> place",
            "roles": ['manufacturer', 'artifact', 'instrument', 'place']
        },
        "manufacture.artifact.n/a": {
            "template": "<arg1> manufactured or created or produced <arg2> using <arg3> at <arg4> place",
            "roles": ['manufacturer', 'artifact', 'instrument', 'place']
        },
        "movement.transportartifact.bringcarryunload": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'artifact', 'vehicle', 'origin', 'destination']
        },
        "movement.transportartifact.disperseseparate": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'artifact', 'vehicle', 'origin', 'destination']
        },
        "movement.transportartifact.fall": {
            "template": "<arg1> fell from <arg2> place to <arg3> place",
            "roles": ['artifact', 'origin', 'destination']
        },
        "movement.transportartifact.grantentry": {
            "template": "<arg1> grants <arg2> entry to <arg3> place from <arg4> place",
            "roles": ['transporter', 'artifact', 'origin', 'destination']
        },
        "movement.transportartifact.hide": {
            "template": "<arg1> concealed <arg2> in <arg3> place, transported in <arg4> vehicle from <arg5> place",
            "roles": ['transporter', 'artifact', 'hidingplace', 'vehicle', 'origin']
        },
        "movement.transportartifact.n/a": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'artifact', 'vehicle', 'origin', 'destination']
        },
        "movement.transportartifact.nonviolentthrowlaunch": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'artifact', 'vehicle', 'origin', 'destination']
        },
        "movement.transportartifact.prevententry": {
            "template": "<arg1> prevents <arg2> from transporting <arg3> from <arg4> place to <arg5> place",
            "roles": ['preventer', 'transporter', 'artifact', 'origin', 'destination']
        },
        "movement.transportartifact.preventexit": {
            "template": "<arg1> prevents <arg2> from transporting <arg3> from <arg4> place to <arg5> place",
            "roles": ['preventer', 'transporter', 'artifact', 'origin', 'destination']
        },
        "movement.transportartifact.receiveimport": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'artifact', 'vehicle', 'origin', 'destination']
        },
        "movement.transportartifact.sendsupplyexport": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'artifact', 'vehicle', 'origin', 'destination']
        },
        "movement.transportartifact.smuggleextract": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'artifact', 'vehicle', 'origin', 'destination']
        },
        "movement.transportperson.bringcarryunload": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'passenger', 'vehicle', 'origin', 'destination']
        },
        "movement.transportperson.disperseseparate": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'passenger', 'vehicle', 'origin', 'destination']
        },
        "movement.transportperson.evacuationrescue": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'passenger', 'vehicle', 'origin', 'destination']
        },
        "movement.transportperson.fall": {
            "template": "<arg1> fell from <arg2> place to <arg3> place",
            "roles": ['passenger', 'origin', 'destination']
        },
        "movement.transportperson.grantentryasylum": {
            "template": "<arg1> grants entry to <arg2> transporting <arg3> from <arg4> place to <arg5> place",
            "roles": ['granter', 'transporter', 'passenger', 'origin', 'destination']
        },
        "movement.transportperson.hide": {
            "template": "<arg1> concealed <arg2> in <arg3> place, transported in <arg4> vehicle from <arg5> place",
            "roles": ['transporter', 'passenger', 'hidingplace', 'vehicle', 'origin']
        },
        "movement.transportperson.n/a": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'passenger', 'vehicle', 'origin', 'destination']
        },
        "movement.transportperson.prevententry": {
            "template": "<arg1> prevents <arg2> from transporting <arg3> from <arg4> place to <arg5> place",
            "roles": ['preventer', 'transporter', 'passenger', 'origin', 'destination']
        },
        "movement.transportperson.preventexit": {
            "template": "<arg1> prevents <arg2> from transporting <arg3> from <arg4> place to <arg5> place",
            "roles": ['preventer', 'transporter', 'passenger', 'origin', 'destination']
        },
        "movement.transportperson.selfmotion": {
            "template": "<arg1> moved in <arg2> from <arg3> place to <arg4> place",
            "roles": ['transporter', 'vehicle', 'origin', 'destination']
        },
        "movement.transportperson.smuggleextract": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['transporter', 'passenger', 'vehicle', 'origin', 'destination']
        },
        "personnel.elect.n/a": {
            "template": "<arg1> elected <arg2> in <arg3> place",
            "roles": ['voter', 'candidate', 'place']
        },
        "personnel.elect.winelection": {
            "template": "<arg1> elected <arg2> in <arg3> place",
            "roles": ['voter', 'candidate', 'place']
        },
        "personnel.endposition.firinglayoff": {
            "template": "<arg1> stopped working at <arg2> in <arg3> place",
            "roles": ['employee', 'placeofemployment', 'place']
        },
        "personnel.endposition.n/a": {
            "template": "<arg1> stopped working at <arg2> in <arg3> place",
            "roles": ['employee', 'placeofemployment', 'place']
        },
        "personnel.endposition.quitretire": {
            "template": "<arg1> stopped working at <arg2> in <arg3> place",
            "roles": ['employee', 'placeofemployment', 'place']
        },
        "personnel.startposition.hiring": {
            "template": "<arg1> started working at <arg2> in <arg3> place",
            "roles": ['employee', 'placeofemployment', 'place']
        },
        "personnel.startposition.n/a": {
            "template": "<arg1> started working at <arg2> in <arg3> place",
            "roles": ['employee', 'placeofemployment', 'place']
        },
        "transaction.transaction.embargosanction": {
            "template": "<arg1> prevented <arg2> from giving <arg4> to <arg3> at <arg5> place",
            "roles": ['preventer', 'giver', 'recipient', 'artifactmoney', 'place']
        },
        "transaction.transaction.giftgrantprovideaid": {
            "template": "<arg1> gave something to <arg2> for the benefit of <arg3> at <arg4> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'place']
        },
        "transaction.transaction.n/a": {
            "template": "A transaction occurred between <arg1> and <arg2> for the benefit of <arg3> at <arg4> place",
            "roles": ['participant', 'participant', 'beneficiary', 'place']
        },
        "transaction.transaction.transfercontrol": {
            "template": "<arg1> transferred control of <arg4> to <arg2> for the benefit of <arg3> in <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'territoryorfacility', 'place']
        },
        "transaction.transfermoney.borrowlend": {
            "template": "<arg1> gave <arg4> money to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'money', 'place']
        },
        "transaction.transfermoney.embargosanction": {
            "template": "<arg1> prevented <arg2> from giving <arg4> to <arg3> at <arg5> place",
            "roles": ['preventer', 'giver', 'recipient', 'money', 'place']
        },
        "transaction.transfermoney.giftgrantprovideaid": {
            "template": "<arg1> gave <arg4> money to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'money', 'place']
        },
        "transaction.transfermoney.n/a": {
            "template": "<arg1> gave <arg4> money to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'money', 'place']
        },
        "transaction.transfermoney.payforservice": {
            "template": "<arg1> gave <arg4> money to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'money', 'place']
        },
        "transaction.transfermoney.purchase": {
            "template": "<arg1> gave <arg4> money to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'money', 'place']
        },
        "transaction.transferownership.borrowlend": {
            "template": "<arg1> gave <arg4> to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'artifact', 'place']
        },
        "transaction.transferownership.embargosanction": {
            "template": "<arg1> prevented <arg2> from giving <arg4> to <arg3> at <arg5> place",
            "roles": ['preventer', 'giver', 'recipient', 'artifact', 'place']
        },
        "transaction.transferownership.giftgrantprovideaid": {
            "template": "<arg1> gave <arg4> to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'artifact', 'place']
        },
        "transaction.transferownership.n/a": {
            "template": "<arg1> gave <arg4> to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'artifact', 'place']
        },
        "transaction.transferownership.purchase": {
            "template": "<arg1> gave <arg4> to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['giver', 'recipient', 'beneficiary', 'artifact', 'place']
        },
    },
    "wikievents": {
        "ArtifactExistence.DamageDestroyDisableDismantle.Damage": {
            "template": "<arg1> damaged <arg2> using <arg3> instrument in <arg4> place",
            "roles": ['Damager', 'Artifact', 'Instrument', 'Place'],
        },
        "ArtifactExistence.DamageDestroyDisableDismantle.Destroy": {
            "template": "<arg1> destroyed <arg2> using <arg3> instrument in <arg4> place",
            "roles": ['Destroyer', 'Artifact', 'Instrument', 'Place'],
        },
        "ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse": {
            "template": "<arg1> disabled or defused <arg2> using <arg3> instrument in <arg4> place",
            "roles": ['Disabler', 'Artifact', 'Instrument', 'Place'],
        },
        "ArtifactExistence.DamageDestroyDisableDismantle.Dismantle": {
            "template": "<arg1> dismantled <arg2> using <arg3> instrument in <arg4> place",
            "roles": ['Dismantler', 'Artifact', 'Instrument', 'Components', 'Place'],
        },
        "ArtifactExistence.DamageDestroyDisableDismantle.Unspecified": {
            "template": "<arg1> damaged or destroyed <arg2> using <arg3> instrument in <arg4> place",
            "roles": ['DamagerDestroyer', 'Artifact', 'Instrument', 'Place'],
        },
        "ArtifactExistence.ManufactureAssemble.Unspecified": {
            "template": "<arg1> manufactured or assembled or produced <arg2> from <arg3> components using <arg4> at <arg5> place",
            "roles": ['ManufacturerAssembler', 'Artifact', 'Components', 'Instrument', 'Place'],
        },
        "Cognitive.IdentifyCategorize.Unspecified": {
            "template": "<arg1> identified <arg2> as <arg3> at <arg4> place",
            "roles": ['Identifier', 'IdentifiedObject', 'IdentifiedRole', 'Place'],
        },
        "Cognitive.Inspection.SensoryObserve": {
            "template": "<arg1> observed <arg2> using <arg3> in <arg4> place",
            "roles": ['Observer', 'ObservedEntity', 'Instrument', 'Place'],
        },
        "Cognitive.Research.Unspecified": {
            "template": "<arg1> researched <arg2> subject using <arg3> at <arg4> place",
            "roles": ['Researcher', 'Subject', 'Means', 'Place'],
        },
        "Cognitive.TeachingTrainingLearning.Unspecified": {
            "template": "<arg1> taught <arg2> field of knowledge to <arg3> using <arg4> at <arg5> institution in <arg6> place",
            "roles": ['TeacherTrainer', 'FieldOfKnowledge', 'Learner', 'Means', 'Institution', 'Place'],
        },
        "Conflict.Attack.DetonateExplode": {
            "template": "<arg1> detonated or exploded <arg4> explosive device using <arg3> to attack <arg2> target at <arg5> place",
            "roles": ['Attacker', 'Target', 'Instrument', 'ExplosiveDevice', 'Place'],
        },
        "Conflict.Attack.Unspecified": {
            "template": "<arg1> attacked <arg2> using <arg3> at <arg4> place",
            "roles": ['Attacker', 'Target', 'Instrument', 'Place'],
        },
        "Conflict.Defeat.Unspecified": {
            "template": "<arg1> defeated <arg2> in <arg3> conflict at <arg4> place",
            "roles": ['Victor', 'Defeated', 'ConflictOrElection', 'Place'],
        },
        "Conflict.Demonstrate.DemonstrateWithViolence": {
            "template": "<arg1> was in a demonstration involving violence for <arg4> topic with <arg3> visual display against <arg5> at <arg6> place, with potential involvement of <arg2> police or military",
            "roles": ['Demonstrator', 'Regulator', 'VisualDisplay', 'Topic', 'Target', 'Place'],
        },
        "Conflict.Demonstrate.Unspecified": {
            "template": "<arg1> was in a demonstration for <arg4> topic with <arg3> visual display against <arg5> at <arg6> place, with potential involvement of <arg2> police or military",
            "roles": ['Demonstrator', 'Regulator', 'VisualDisplay', 'Topic', 'Target', 'Place'],
        },
        "Contact.Contact.Broadcast": {
            "template": "<arg1> communicated to <arg2> about <arg4> topic using <arg3> at <arg5> place (one-way communication)",
            "roles": ['Communicator', 'Recipient', 'Instrument', 'Topic', 'Place'],
        },
        "Contact.Contact.Correspondence": {
            "template": "<arg1> communicated remotely with <arg2> about <arg4> topic using <arg3> at <arg5> place",
            "roles": ['Participant', 'Participant', 'Instrument', 'Topic', 'Place'],
        },
        "Contact.Contact.Meet": {
            "template": "<arg1> met face-to-face with <arg2> about <arg3> topic at <arg4> place",
            "roles": ['Participant', 'Participant', 'Topic', 'Place'],
        },
        "Contact.Contact.Unspecified": {
            "template": "<arg1> communicated with <arg2> about <arg3> topic at <arg4> place (document does not specify in person or not, or one-way or not)",
            "roles": ['Participant', 'Participant', 'Topic', 'Place'],
        },
        "Contact.RequestCommand.Broadcast": {
            "template": "<arg1> communicated with <arg2> about <arg3> topic at <arg4> place (document does not specify in person or not, or one-way or not)",
            "roles": ['Communicator', 'Recipient', 'Topic', 'Place'],
        },
        "Contact.RequestCommand.Correspondence": {
            "template": "<arg1> communicated with <arg2> about <arg3> topic at <arg4> place (document does not specify in person or not, or one-way or not)",
            "roles": ['Communicator', 'Recipient', 'Topic', 'Place'],
        },
        "Contact.RequestCommand.Meet": {
            "template": "<arg1> communicated with <arg2> about <arg3> topic at <arg4> place (document does not specify in person or not, or one-way or not)",
            "roles": ['Communicator', 'Recipient', 'Topic', 'Place'],
        },
        "Contact.RequestCommand.Unspecified": {
            "template": "<arg1> communicated with <arg2> about <arg3> topic at <arg4> place (document does not specify in person or not, or one-way or not)",
            "roles": ['Communicator', 'Recipient', 'Topic', 'Place'],
        },
        "Contact.ThreatenCoerce.Broadcast": {
            "template": "<arg1> communicated with <arg2> about <arg3> topic at <arg4> place (document does not specify in person or not, or one-way or not)",
            "roles": ['Communicator', 'Recipient', 'Topic', 'Place'],
        },
        "Contact.ThreatenCoerce.Correspondence": {
            "template": "<arg1> communicated with <arg2> about <arg3> topic at <arg4> place (document does not specify in person or not, or one-way or not)",
            "roles": ['Communicator', 'Recipient', 'Topic', 'Place'],
        },
        "Contact.ThreatenCoerce.Unspecified": {
            "template": "<arg1> communicated with <arg2> about <arg3> topic at <arg4> place (document does not specify in person or not, or one-way or not)",
            "roles": ['Communicator', 'Recipient', 'Topic', 'Place'],
        },
        "Control.ImpedeInterfereWith.Unspecified": {
            "template": "<arg1> impeded or interfered with <arg2> event at <arg3> place",
            "roles": ['Impeder', 'ImpededEvent', 'Place'],
        },
        "Disaster.Crash.Unspecified": {
            "template": "<arg1> person in <arg2> vehicle crashed into <arg3> at <arg4> place",
            "roles": ['DriverPassenger', 'Vehicle', 'CrashObject', 'Place'],
        },
        "Disaster.DiseaseOutbreak.Unspecified": {
            "template": "<arg1> disease broke out among <arg2> victims or population at <arg3> place",
            "roles": ['Disease', 'Victim', 'Place'],
        },
        "GenericCrime.GenericCrime.GenericCrime": {
            "template": "<arg1> committed a crime against <arg2> at <arg3> place",
            "roles": ['Perpetrator', 'Victim', 'Place'],
        },
        "Justice.Acquit.Unspecified": {
            "template": "<arg1> court or judge acquitted <arg2> of <arg3> crime in <arg4> place",
            "roles": ['JudgeCourt', 'Defendant', 'Crime', 'Place'],
        },
        "Justice.ArrestJailDetain.Unspecified": {
            "template": "<arg1> arrested or jailed <arg2> for <arg3> crime at <arg4> place",
            "roles": ['Jailer', 'Detainee', 'Crime', 'Place'],
        },
        "Justice.ChargeIndict.Unspecified": {
            "template": "<arg1> charged or indicted <arg2> before <arg3> court or judge for <arg4> crime in <arg5> place",
            "roles": ['Prosecutor', 'Defendant', 'JudgeCourt', 'Crime', 'Place'],
        },
        "Justice.Convict.Unspecified": {
            "template": "<arg1> court or judge convicted <arg2> of <arg3> crime in <arg4> place",
            "roles": ['JudgeCourt', 'Defendant', 'Crime', 'Place'],
        },
        "Justice.InvestigateCrime.Unspecified": {
            "template": "<arg1> investigated <arg2> for <arg3> crime in <arg4> place",
            "roles": ['Investigator', 'Defendant', 'Crime', 'Place'],
        },
        "Justice.ReleaseParole.Unspecified": {
            "template": "<arg1> court or judge released or paroled <arg2> from <arg3> crime in <arg4> place",
            "roles": ['JudgeCourt', 'Defendant', 'Crime', 'Place'],
        },
        "Justice.Sentence.Unspecified": {
            "template": "<arg1> court or judge sentenced <arg2> for <arg3> crime to <arg4> sentence in <arg5> place",
            "roles": ['JudgeCourt', 'Defendant', 'Crime', 'Sentence', 'Place'],
        },
        "Justice.TrialHearing.Unspecified": {
            "template": "<arg1> tried <arg2> before <arg3> court or judge for <arg4> crime in <arg5> place",
            "roles": ['Prosecutor', 'Defendant', 'JudgeCourt', 'Crime', 'Place'],
        },
        "Life.Die.Unspecified": {
            "template": "<arg1> died at <arg2> place from <arg4> medical issue, killed by <arg3> killer",
            "roles": ['Victim', 'Place', 'Killer', 'MedicalIssue'],
        },
        "Life.Infect.Unspecified": {
            "template": "<arg1> was infected with <arg2> from <arg3> at <arg4> place",
            "roles": ['Victim', 'InfectingAgent', 'Source', 'Place'],
        },
        "Life.Injure.Unspecified": {
            "template": "<arg1> was injured by <arg2> using <arg3> in <arg4> body part with <arg5> medical issue at <arg6> place",
            "roles": ['Victim', 'Injurer', 'Instrument', 'BodyPart', 'MedicalCondition', 'Place'],
        },
        "Medical.Intervention.Unspecified": {
            "template": "<arg1> treater treated <arg2> patient for <arg3> medical issue with <arg4> means at <arg5> place",
            "roles": ['Treater', 'Patient', 'MedicalIssue', 'Instrument', 'Place'],
        },
        "Movement.Transportation.Evacuation": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['Transporter', 'PassengerArtifact', 'Vehicle', 'Origin', 'Destination'],
        },
        "Movement.Transportation.IllegalTransportation": {
            "template": "<arg1> illegally transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['Transporter', 'PassengerArtifact', 'Vehicle', 'Origin', 'Destination'],
        },
        "Movement.Transportation.PreventPassage": {
            "template": "<arg4> prevents <arg1> from entering <arg6> place from <arg5> place to transport <arg2> using <arg3> vehicle",
            "roles": ['Transporter', 'PassengerArtifact', 'Vehicle', 'Preventer', 'Origin', 'Destination'],
        },
        "Movement.Transportation.Unspecified": {
            "template": "<arg1> transported <arg2> in <arg3> from <arg4> place to <arg5> place",
            "roles": ['Transporter', 'PassengerArtifact', 'Vehicle', 'Origin', 'Destination'],
        },
        "Personnel.EndPosition.Unspecified": {
            "template": "<arg1> stopped working in <arg3> position at <arg2> organization in <arg4> place",
            "roles": ['Employee', 'PlaceOfEmployment', 'Position', 'Place'],
        },
        "Personnel.StartPosition.Unspecified": {
            "template": "<arg1> started working in <arg3> position at <arg2> organization in <arg4> place",
            "roles": ['Employee', 'PlaceOfEmployment', 'Position', 'Place'],
        },
        "Transaction.Donation.Unspecified": {
            "template": "<arg1> gave <arg4> to <arg2> for the benefit of <arg3> at <arg5> place",
            "roles": ['Giver', 'Recipient', 'Beneficiary', 'ArtifactMoney', 'Place'],
        },
        "Transaction.ExchangeBuySell.Unspecified": {
            "template": "<arg1> bought, sold, or traded <arg3> to <arg2> in exchange for <arg4> for the benefit of <arg5> at <arg6> place",
            "roles": ['Giver', 'Recipient', 'AcquiredEntity', 'PaymentBarter', 'Beneficiary', 'Place'],
        },
    },
    "phee": {
        "Adverse_event": {
            "template": "<arg1> Combination Drug <arg2> Effect <arg3> Subject <arg4> Subject Age <arg5> Subject Disorder <arg6> Subject Gender <arg7> Subject Population <arg8> Subject Race <arg9> Treatment <arg10> Treatment Disorder <arg11> Treatment Dosage <arg12> Treatment Drug <arg13> Treatment Duration <arg14> Treatment Frequency <arg15> Treatment Route <arg16> Treatment Time Elapsed",
            "roles": [
                "Combination_Drug",
                "Effect",
                "Subject",
                "Subject_Age",
                "Subject_Disorder",
                "Subject_Gender",
                "Subject_Population",
                "Subject_Race",
                "Treatment",
                "Treatment_Disorder",
                "Treatment_Dosage",
                "Treatment_Drug",
                "Treatment_Duration",
                "Treatment_Freq",
                "Treatment_Route",
                "Treatment_Time_elapsed",
            ],
        },
        "Potential_therapeutic_event": {
            "template": "<arg1> Combination Drug <arg2> Effect <arg3> Subject <arg4> Subject Age <arg5> Subject Disorder <arg6> Subject Gender <arg7> Subject Population <arg8> Subject Race <arg9> Treatment <arg10> Treatment Disorder <arg11> Treatment Dosage <arg12> Treatment Drug <arg13> Treatment Duration <arg14> Treatment Frequency <arg15> Treatment Route <arg16> Treatment Time Elapsed",
            "roles": [
                "Combination_Drug",
                "Effect",
                "Subject",
                "Subject_Age",
                "Subject_Disorder",
                "Subject_Gender",
                "Subject_Population",
                "Subject_Race",
                "Treatment",
                "Treatment_Disorder",
                "Treatment_Dosage",
                "Treatment_Drug",
                "Treatment_Duration",
                "Treatment_Freq",
                "Treatment_Route",
                "Treatment_Time_elapsed",
            ],
        },
    },
    "docee": {
        "Air Crash": {
            "template": "<arg1> accident investigator <arg2> aircraft agency <arg3> alternate landing place <arg4> casualties and losses <arg5> cause <arg6> crew <arg7> date <arg8> flight Number <arg9> location <arg10> passengers <arg11> scheduled landing place <arg12> service years <arg13> survivors <arg14> taking-off place",
            "roles": ['Accident Investigator', 'Aircraft Agency', 'Alternate Landing Place', 'Casualties and Losses', 'Cause', 'Crew', 'Date', 'Flight Number', 'Location', 'Passengers', 'Scheduled Landing Place', 'Service Years', 'Survivors', 'Taking-off Place'],
        },
        "Appoint_Inauguration": {
            "template": "<arg1> age of the appointee <arg2> appointee <arg3> appointer <arg4> appointment time <arg5> employment agency <arg6> inauguration time <arg7> last job of the appointee <arg8> position <arg9> predecessor <arg10> term of office",
            "roles": ['Age of the Appointee', 'Appointee', 'Appointer', 'Appointment Time', 'Employment Agency', 'Inauguration Time', 'Last Job of the Appointee', 'Position', 'Predecessor', 'Term of Office'],
        },
        "Armed Conflict": {
            "template": "<arg1> attacker <arg2> casualties and losses <arg3> conflict duration <arg4> damaged facility <arg5> date <arg6> end time <arg7> location <arg8> military strength <arg9> target <arg10> weapon and equippment",
            "roles": ['Attacker', 'Casualties and Losses', 'Conflict Duration', 'Damaged Facility', 'Date', 'End Time', 'Location', 'Military Strength', 'Target', 'Weapon and Equippment'],
        },
        "Awards Ceremony": {
            "template": "<arg1> award <arg2> award field <arg3> award reason <arg4> date <arg5> employed institution <arg6> host <arg7> location <arg8> winner",
            "roles": ['Award', 'Award Field', 'Award Reason', 'Date', 'Employed Institution', 'Host', 'Location', 'Winner'],
        },
        "Bank Robbery": {
            "template": "<arg1> arrested <arg2> bank name <arg3> date <arg4> hostage <arg5> investigating spokesperson <arg6> investigating agency <arg7> location <arg8> perpetrators <arg9> recovered amount <arg10> stolen amount <arg11> transportation <arg12> weapon used",
            "roles": ['Arrested', 'Bank Name', 'Date', 'Hostage', 'Investigating Spokesperson', 'Investigating agency', 'Location', 'Perpetrators', 'Recovered Amount', 'Stolen Amount', 'Transportation', 'Weapon Used'],
        },
        "Break Historical Records": {
            "template": "<arg1> date <arg2> grades <arg3> last time the record was broken <arg4> location <arg5> previous record holder <arg6> record breaker <arg7> record-breaking project <arg8> the grades of the previous record holder",
            "roles": ['Date', 'Grades', 'Last Time the Record was Broken', 'Location', 'Previous Record Holder', 'Record Breaker', 'Record-breaking Project', 'The Grades of the Previous Record Holder'],
        },
        "CommitCrime - Accuse": {
            "template": "<arg1> accused people <arg2> charged crime <arg3> defense lawyer <arg4> evidence <arg5> prosecution lawyer <arg6> prosecutor <arg7> time of the case <arg8> witness",
            "roles": ['Accused People', 'Charged Crime', 'Defense Lawyer', 'Evidence', 'Prosecution Lawyer', 'Prosecutor', 'Time of the Case', 'Witness'],
        },
        "CommitCrime - Arrest": {
            "template": "<arg1> arrest location <arg2> arrest time <arg3> criminal evidence <arg4> police <arg5> suspect <arg6> the charged crime",
            "roles": ['Arrest Location', 'Arrest Time', 'Criminal Evidence', 'Police', 'Suspect', 'The Charged Crime'],
        },
        "CommitCrime - Investigate": {
            "template": "<arg1> cause <arg2> date <arg3> head of investigation team <arg4> investigative agency <arg5> person under investigation",
            "roles": ['Cause', 'Date', 'Head of Investigation Team', 'Investigative Agency', 'Person under Investigation'],
        },
        "CommitCrime - Release": {
            "template": "<arg1> charged crime <arg2> defense lawyer <arg3> jail time <arg4> judge <arg5> prison term <arg6> release reason <arg7> release time <arg8> released people <arg9> sentencing location",
            "roles": ['Charged Crime', 'Defense Lawyer', 'Jail Time', 'Judge', 'Prison Term', 'Release Reason', 'Release Time', 'Released People', 'Sentencing Location'],
        },
        "CommitCrime - Sentence": {
            "template": "<arg1> accusation <arg2> court <arg3> court time <arg4> defense lawyer <arg5> detention start time <arg6> judge <arg7> judgement result/prison term <arg8> prison <arg9> prosecution lawyer <arg10> release time <arg11> suspect <arg12> the sentence claimed by the prosecutor's lawyer <arg13> victim",
            "roles": ['Accusation', 'Court', 'Court Time', 'Defense Lawyer', 'Detention Start Time', 'Judge', 'Judgement Result/Prison Term', 'Prison', 'Prosecution Lawyer', 'Release Time', 'Suspect', "The Sentence Claimed by the Prosecutor's Lawyer", 'Victim'],
        },
        "Diplomatic Talks": {
            "template": "<arg1> achievement <arg2> date <arg3> location <arg4> participants <arg5> summit name <arg6> summit theme",
            "roles": ['Achievement', 'Date', 'Location', 'Participants', 'Summit Name', 'Summit Theme'],
        },
        "Diplomatic Visit": {
            "template": "<arg1> country visited <arg2> date <arg3> host <arg4> visitor",
            "roles": ['Country Visited', 'Date', 'Host', 'Visitor'],
        },
        "Disease Outbreaks": {
            "template": "<arg1> areas affected <arg2> cause <arg3> complications <arg4> confirmed/infected cases <arg5> cured cases <arg6> cured rate <arg7> date <arg8> death cases <arg9> death rate <arg10> disease <arg11> epidemic data issuing agency <arg12> number of people hospitalized <arg13> number of vaccinated people <arg14> outbreak location <arg15> precautionary measure <arg16> special medicine <arg17> susceptible population <arg18> suspected cases <arg19> symptoms <arg20> vaccine research and development organization <arg21> way for spreading",
            "roles": ['Areas affected', 'Cause', 'Complications', 'Confirmed/Infected Cases', 'Cured Cases', 'Cured Rate', 'Date', 'Death Cases', 'Death Rate', 'Disease', 'Epidemic Data Issuing Agency', 'Number of People Hospitalized', 'Number of Vaccinated People', 'Outbreak Location', 'Precautionary Measure', 'Special Medicine', 'Susceptible Population', 'Suspected Cases', 'Symptoms', 'Vaccine Research and Development Organization', 'Way for Spreading'],
        },
        "Droughts": {
            "template": "<arg1> areas affected <arg2> cause <arg3> damaged crops & livestock <arg4> date <arg5> economic loss <arg6> influenced people <arg7> production cuts <arg8> related rivers or lakes <arg9> the worst-hit area",
            "roles": ['Areas Affected', 'Cause', 'Damaged Crops & Livestock', 'Date', 'Economic Loss', 'Influenced People', 'Production Cuts', 'Related Rivers or Lakes', 'The Worst-Hit Area'],
        },
        "Earthquakes": {
            "template": "<arg1> affected area <arg2> aid agency <arg3> aid supplies/amount <arg4> casualties and losses <arg5> date <arg6> economic loss <arg7> epicenter <arg8> magnitude <arg9> number of aftershocks <arg10> number of destroyed building <arg11> number of evacuated people <arg12> number of rescued people <arg13> number of trapped people <arg14> temporary settlement",
            "roles": ['Affected Area', 'Aid Agency', 'Aid Supplies/Amount', 'Casualties and Losses', 'Date', 'Economic Loss', 'Epicenter', 'Magnitude', 'Number of Aftershocks', 'Number of Destroyed Building', 'Number of Evacuated People', 'Number of Rescued People', 'Number of Trapped People', 'Temporary Settlement'],
        },
        "Election": {
            "template": "<arg1> candidate scandal(allegations of fraud, etc) <arg2> candidates and their political parties <arg3> date <arg4> election goal <arg5> election name <arg6> electoral system <arg7> location <arg8> people casting key votes <arg9> the final seats result <arg10> the final votes and percentages <arg11> turnout(who win who lost) <arg12> voting method",
            "roles": ['Candidate Scandal(Allegations of fraud, etc)', 'Candidates and their Political Parties', 'Date', 'Election Goal', 'Election Name', 'Electoral System', 'Location', 'People Casting Key Votes', 'The Final Seats Result', 'The Final Votes and Percentages', 'Turnout(who win who lost)', 'Voting method'],
        },
        "Environment Pollution": {
            "template": "<arg1> anti-pollution people/organizations <arg2> cause <arg3> date <arg4> location <arg5> number of victims <arg6> party responsible for pollution <arg7> pollution source <arg8> solution",
            "roles": ['Anti-pollution People/Organizations', 'Cause', 'Date', 'Location', 'Number of Victims', 'Party Responsible for Pollution', 'Pollution Source', 'Solution'],
        },
        "Famine": {
            "template": "<arg1> affected areas <arg2> aid agency <arg3> aid supplies/amount <arg4> casualities and losses <arg5> cause <arg6> date <arg7> number of influenced people <arg8> solution",
            "roles": ['Affected Areas', 'Aid Agency', 'Aid Supplies/Amount', 'Casualities and Losses', 'Cause', 'Date', 'Number of Influenced People', 'Solution'],
        },
        "Famous Person - Death": {
            "template": "<arg1> age <arg2> date <arg3> death reason <arg4> deceased <arg5> doctor <arg6> location/hospital <arg7> perpetrator <arg8> profession <arg9> state before death",
            "roles": ['Age', 'Date', 'Death Reason', 'Deceased', 'Doctor', 'Location/Hospital', 'Perpetrator', 'Profession', 'State Before Death'],
        },
        "Famous Person - Divorce": {
            "template": "<arg1> announce platform <arg2> cause <arg3> child <arg4> child custody <arg5> date <arg6> husband <arg7> marriage duration <arg8> property division <arg9> wife",
            "roles": ['Announce Platform', 'Cause', 'Child', 'Child Custody', 'Date', 'Husband', 'Marriage Duration', 'Property Division', 'Wife'],
        },
        "Famous Person - Give a Speech": {
            "template": "<arg1> date <arg2> location <arg3> news release agency <arg4> speaker <arg5> speaker status <arg6> ways to watch the speech",
            "roles": ['Date', 'Location', 'News Release Agency', 'Speaker', 'Speaker Status', 'Ways to Watch the Speech'],
        },
        "Famous Person - Marriage": {
            "template": "<arg1> cost <arg2> date <arg3> how many times get married <arg4> husband <arg5> invited person <arg6> wedding dress designer <arg7> wedding venue <arg8> wife <arg9> witness",
            "roles": ['Cost', 'Date', 'How Many Times Get Married', 'Husband', 'Invited Person', 'Wedding Dress Designer', 'Wedding Venue', 'Wife', 'Witness'],
        },
        "Famous Person - Recovered": {
            "template": "<arg1> date <arg2> disease <arg3> doctor and medical team <arg4> location/hospital <arg5> people <arg6> sequelae <arg7> treatment method",
            "roles": ['Date', 'Disease', 'Doctor and Medical Team', 'Location/Hospital', 'People', 'Sequelae', 'Treatment Method'],
        },
        "Famous Person - Sick": {
            "template": "<arg1> cause <arg2> date <arg3> doctor and medical team <arg4> illness <arg5> location/hospital <arg6> people <arg7> symptom",
            "roles": ['Cause', 'Date', 'Doctor and Medical Team', 'Illness', 'Location/Hospital', 'People', 'Symptom'],
        },
        "Financial Aid": {
            "template": "<arg1> aid reason <arg2> beneficiary <arg3> date <arg4> funding <arg5> location <arg6> sponsor",
            "roles": ['Aid Reason', 'Beneficiary', 'Date', 'Funding', 'Location', 'Sponsor'],
        },
        "Financial Crisis": {
            "template": "<arg1> affected area <arg2> affected industries <arg3> bankrupt business <arg4> cause <arg5> economic loss <arg6> economists who predicted the crisis <arg7> end_date <arg8> policy proposals <arg9> start_date <arg10> unemployed rate",
            "roles": ['Affected Area', 'Affected Industries', 'Bankrupt Business', 'Cause', 'Economic Loss', 'Economists who Predicted the Crisis', 'End_Date', 'Policy Proposals', 'Start_Date', 'Unemployed Rate'],
        },
        "Fire": {
            "template": "<arg1> aid agency <arg2> aid supplies/amount <arg3> casualties and losses <arg4> cause <arg5> date <arg6> economic loss <arg7> location <arg8> magnitude <arg9> number of damaged houses <arg10> number of rescued people <arg11> temporary settlement",
            "roles": ['Aid Agency', 'Aid Supplies/Amount', 'Casualties and Losses', 'Cause', 'Date', 'Economic Loss', 'Location', 'Magnitude', 'Number of Damaged Houses', 'Number of Rescued People', 'Temporary Settlement'],
        },
        "Floods": {
            "template": "<arg1> affected areas <arg2> aid agency <arg3> aid supplies/amount <arg4> casualties and losses <arg5> cause <arg6> date <arg7> disaster-stricken farmland <arg8> economic loss <arg9> maximum rainfall <arg10> missings <arg11> number of damaged houses <arg12> number of evacuated people <arg13> number of rescued people <arg14> related rivers or lakes <arg15> temporary settlement <arg16> water level",
            "roles": ['Affected Areas', 'Aid Agency', 'Aid Supplies/Amount', 'Casualties and Losses', 'Cause', 'Date', 'Disaster-Stricken Farmland', 'Economic Loss', 'Maximum Rainfall', 'Missings', 'Number of Damaged Houses', 'Number of Evacuated People', 'Number of Rescued People', 'Related Rivers or Lakes', 'Temporary Settlement', 'Water Level'],
        },
        "Gas Explosion": {
            "template": "<arg1> accident investigator <arg2> attending hospital <arg3> casualties and losses <arg4> cause <arg5> compensation <arg6> date <arg7> economic loss <arg8> location <arg9> survivors",
            "roles": ['Accident Investigator', 'Attending Hospital', 'Casualties and Losses', 'Cause', 'Compensation', 'Date', 'Economic loss', 'Location', 'Survivors'],
        },
        "Government Policy Changes": {
            "template": "<arg1> announcement date <arg2> bill drafting agency <arg3> cause <arg4> deliberating agency <arg5> effective date <arg6> invalid date <arg7> location <arg8> policy content <arg9> policy name & abbreviation",
            "roles": ['Announcement Date', 'Bill Drafting Agency', 'Cause', 'Deliberating Agency', 'Effective Date', 'Invalid Date', 'Location', 'Policy Content', 'Policy Name & Abbreviation'],
        },
        "Insect Disaster": {
            "template": "<arg1> affected areas <arg2> aid agency <arg3> aid supplies/amount <arg4> cause <arg5> date <arg6> economic loss <arg7> influenced crops and livelihood <arg8> pests <arg9> response measures",
            "roles": ['Affected Areas', 'Aid Agency', 'Aid Supplies/Amount', 'Cause', 'Date', 'Economic Loss', 'Influenced Crops and Livelihood', 'Pests', 'Response Measures'],
        },
        "Join in an Orgnization": {
            "template": "<arg1> annoncement date <arg2> cause <arg3> countries joined the organization <arg4> declarer <arg5> effective date <arg6> join conditions <arg7> organization leader <arg8> organization members <arg9> organization name",
            "roles": ['Annoncement Date', 'Cause', 'Countries Joined the Organization', 'Declarer', 'Effective Date', 'Join Conditions', 'Organization Leader', 'Organization Members', 'Organization Name'],
        },
        "Mass Poisoning": {
            "template": "<arg1> accident investigator <arg2> attending hospital <arg3> casualties and losses <arg4> cause <arg5> compensation <arg6> date <arg7> economic loss <arg8> location <arg9> poisoning type",
            "roles": ['Accident Investigator', 'Attending Hospital', 'Casualties and Losses', 'Cause', 'Compensation', 'Date', 'Economic Loss', 'Location', 'Poisoning Type'],
        },
        "Military Excercise": {
            "template": "<arg1> army <arg2> commanders and their position <arg3> date <arg4> goal <arg5> location <arg6> military exercise <arg7> participating countries <arg8> projects <arg9> scale <arg10> weapon and equippment",
            "roles": ['Army', 'Commanders and their Position', 'Date', 'Goal', 'Location', 'Military Exercise', 'Participating Countries', 'Projects', 'Scale', 'Weapon and Equippment'],
        },
        "Mine Collapses": {
            "template": "<arg1> accident investigator <arg2> attending hospital <arg3> casualties and losses <arg4> cause <arg5> date <arg6> economic loss <arg7> location <arg8> number of trapped people <arg9> rescue start time <arg10> survivors <arg11> trapped days <arg12> trapped depth",
            "roles": ['Accident Investigator', 'Attending Hospital', 'Casualties and Losses', 'Cause', 'Date', 'Economic Loss', 'Location', 'Number of Trapped People', 'Rescue Start Time', 'Survivors', 'Trapped Days', 'Trapped Depth'],
        },
        "Mudslides": {
            "template": "<arg1> affected areas <arg2> aid agency <arg3> casualties and losses <arg4> cause <arg5> date <arg6> influence people <arg7> missings <arg8> number of destroyed building <arg9> number of evacuated people <arg10> number of rescued people",
            "roles": ['Affected Areas', 'Aid Agency', 'Casualties and Losses', 'Cause', 'Date', 'Influence People', 'Missings', 'Number of Destroyed Building', 'Number of Evacuated People', 'Number of Rescued People'],
        },
        "New Achievements in Aerospace": {
            "template": "<arg1> astronauts <arg2> carrier rocket <arg3> cooperative agency <arg4> launch country <arg5> launch date <arg6> launch result <arg7> launch site <arg8> mission duration <arg9> research agency <arg10> spacecraft <arg11> spacecraft mission <arg12> spokeswoman/spokesman",
            "roles": ['Astronauts', 'Carrier Rocket', 'Cooperative Agency', 'Launch Country', 'Launch Date', 'Launch Result', 'Launch Site', 'Mission Duration', 'Research Agency', 'Spacecraft', 'Spacecraft Mission', 'Spokeswoman/Spokesman'],
        },
        "New Archeological Discoveries": {
            "template": "<arg1> archaeologist <arg2> archaeologist organization <arg3> artifacts and their chronology <arg4> discover location <arg5> discover time <arg6> historical sites <arg7> reasons for the formation of the historical sites",
            "roles": ['Archaeologist', 'Archaeologist Organization', 'Artifacts and Their Chronology', 'Discover Location', 'Discover Time', 'Historical Sites', 'Reasons for the Formation of the Historical sites'],
        },
        "New Wonders in Nature": {
            "template": "<arg1> best way to shoot <arg2> cause <arg3> forecasting agency <arg4> live broadcast platform <arg5> spectacle duration <arg6> spectacle end time <arg7> spectacle location <arg8> spectacle start time <arg9> types of the spectacle",
            "roles": ['Best Way to Shoot', 'Cause', 'Forecasting Agency', 'Live Broadcast Platform', 'Spectacle Duration', 'Spectacle End Time', 'Spectacle Location', 'Spectacle Start Time', 'Types of the Spectacle'],
        },
        "Organization Closed": {
            "template": "<arg1> cause <arg2> date <arg3> head of the institution <arg4> location <arg5> organization",
            "roles": ['Cause', 'Date', 'Head of the Institution', 'Location', 'Organization'],
        },
        "Organization Established": {
            "template": "<arg1> cause <arg2> date <arg3> head of institution <arg4> location <arg5> organization <arg6> registered capital <arg7> spokenmen",
            "roles": ['Cause', 'Date', 'Head of Institution', 'Location', 'Organization', 'Registered Capital', 'Spokenmen'],
        },
        "Organization Fine": {
            "template": "<arg1> date <arg2> fine reason <arg3> fined agency <arg4> lawyer <arg5> location <arg6> penalty amount <arg7> regulatory authority",
            "roles": ['Date', 'Fine Reason', 'Fined Agency', 'Lawyer', 'Location', 'Penalty Amount', 'Regulatory Authority'],
        },
        "Organization Merge": {
            "template": "<arg1> acquiree <arg2> acquirer <arg3> acquisition amount <arg4> date <arg5> head of the merged organization <arg6> merger terms <arg7> organization industry",
            "roles": ['Acquiree', 'Acquirer', 'Acquisition amount', 'Date', 'Head of the Merged Organization', 'Merger Terms', 'Organization Industry'],
        },
        "Protest": {
            "template": "<arg1> arrested <arg2> casualties and losses <arg3> damaged property <arg4> date <arg5> government reaction <arg6> location <arg7> method <arg8> protest reason <arg9> protest slogan <arg10> protesters",
            "roles": ['Arrested', 'Casualties and Losses', 'Damaged Property', 'Date', 'Government Reaction', 'Location', 'Method', 'Protest Reason', 'Protest Slogan', 'Protesters'],
        },
        "Regime Change": {
            "template": "<arg1> army <arg2> commanders of the army <arg3> country <arg4> date <arg5> head of the government <arg6> provisional government <arg7> time for dignitaries to resign",
            "roles": ['Army', 'Commanders of the Army', 'Country', 'Date', 'Head of the Government', 'Provisional Government', 'Time for Dignitaries to Resign'],
        },
        "Resignation_Dismissal": {
            "template": "<arg1> age of the resignated person <arg2> approver <arg3> date <arg4> employment agency <arg5> position <arg6> resign reason <arg7> resignated person <arg8> successor <arg9> term of office",
            "roles": ['Age of the Resignated Person', 'Approver', 'Date', 'Employment Agency', 'Position', 'Resign Reason', 'Resignated Person', 'Successor', 'Term of Office'],
        },
        "Riot": {
            "template": "<arg1> belligerents <arg2> casualties and losses <arg3> date <arg4> economic loss <arg5> location <arg6> riot reason <arg7> weapon",
            "roles": ['Belligerents', 'Casualties and Losses', 'Date', 'Economic Loss', 'Location', 'Riot Reason', 'Weapon'],
        },
        "Road Crash": {
            "template": "<arg1> accident investigator <arg2> attending hospital <arg3> casualties and losses <arg4> cause <arg5> date <arg6> economic loss <arg7> location <arg8> number of vehicles involved in the crash <arg9> survivors",
            "roles": ['Accident Investigator', 'Attending Hospital', 'Casualties and Losses', 'Cause', 'Date', 'Economic loss', 'Location', 'Number of Vehicles Involved in the Crash', 'Survivors'],
        },
        "Shipwreck": {
            "template": "<arg1> accident investigator <arg2> casualties and losses <arg3> date <arg4> hull discovery time <arg5> hull location <arg6> location <arg7> missings <arg8> rescue organizer <arg9> rescue start time <arg10> rescue tool or method <arg11> ship agency <arg12> ship Number <arg13> shipwreck reason <arg14> state of the hull <arg15> survivors",
            "roles": ['Accident Investigator', 'Casualties and Losses', 'Date', 'Hull Discovery Time', 'Hull Location', 'Location', 'Missings', 'Rescue Organizer', 'Rescue Start Time', 'Rescue Tool or Method', 'Ship Agency', 'Ship Number', 'Shipwreck Reason', 'State of the Hull', 'Survivors'],
        },
        "Sign Agreement": {
            "template": "<arg1> agreement content <arg2> agreement name <arg3> agreement validity period <arg4> contracting parties <arg5> date <arg6> location",
            "roles": ['Agreement Content', 'Agreement Name', 'Agreement Validity Period', 'Contracting Parties', 'Date', 'Location'],
        },
        "Sports Competition": {
            "template": "<arg1> champions <arg2> competition items <arg3> contest participant <arg4> end time <arg5> game name <arg6> host country <arg7> lasting time <arg8> location <arg9> mvp <arg10> postpone reason <arg11> postpone time <arg12> score <arg13> start time",
            "roles": ['Champions', 'Competition Items', 'Contest Participant', 'End Time', 'Game Name', 'Host Country', 'Lasting Time', 'Location', 'MVP', 'Postpone Reason', 'Postpone Time', 'Score', 'Start Time'],
        },
        "Storm": {
            "template": "<arg1> amount of precipitation <arg2> influence people <arg3> maximum wind speed <arg4> people/organization who predicted the disaster <arg5> storm center location <arg6> storm direction <arg7> storm formation location <arg8> storm formation time <arg9> storm hit location <arg10> storm hit time <arg11> storm movement speed <arg12> storm name <arg13> storm warning level",
            "roles": ['Amount of Precipitation', 'Influence People', 'Maximum Wind Speed', 'People/Organization who predicted the disaster', 'Storm Center Location', 'Storm Direction', 'Storm Formation Location', 'Storm Formation Time', 'Storm Hit Location', 'Storm Hit Time', 'Storm Movement Speed', 'Storm Name', 'Storm Warning Level'],
        },
        "Strike": {
            "template": "<arg1> boycotted institutions <arg2> duration <arg3> economy loss <arg4> end date <arg5> start date <arg6> strike agency <arg7> strike industry <arg8> strike outcome <arg9> strike reason <arg10> strikers <arg11> strikers status",
            "roles": ['Boycotted Institutions', 'Duration', 'Economy Loss', 'End Date', 'Start Date', 'Strike Agency', 'Strike Industry', 'Strike Outcome', 'Strike Reason', 'Strikers', 'Strikers Status'],
        },
        "Tear Up Agreement": {
            "template": "<arg1> agreement members <arg2> agreement name <arg3> date <arg4> tear up reason <arg5> the agency who broke the agreement",
            "roles": ['Agreement Members', 'Agreement Name', 'Date', 'Tear Up Reason', 'The Agency Who Broke the Agreement'],
        },
        "Train Collisions": {
            "template": "<arg1> accident investigator <arg2> attending hospital <arg3> casualties and losses <arg4> cause <arg5> date <arg6> economic loss <arg7> location <arg8> survivors <arg9> train agency <arg10> train Number",
            "roles": ['Accident Investigator', 'Attending Hospital', 'Casualties and Losses', 'Cause', 'Date', 'Economic Loss', 'Location', 'Survivors', 'Train Agency', 'Train Number'],
        },
        "Tsunamis": {
            "template": "<arg1> area affected <arg2> casualties and losses <arg3> cause <arg4> date <arg5> economic loss <arg6> magnitude(tsunami heights) <arg7> number of destroyed building <arg8> people/organization who predicted the disaster <arg9> tsunamis <arg10> warning device",
            "roles": ['Area Affected', 'Casualties and Losses', 'Cause', 'Date', 'Economic Loss', 'Magnitude(Tsunami heights)', 'Number of Destroyed Building', 'People/Organization who predicted the disaster', 'Tsunamis', 'Warning Device'],
        },
        "Volcano Eruption": {
            "template": "<arg1> affected areas <arg2> casualties and losses <arg3> cause <arg4> economic loss <arg5> fire warning level <arg6> last outbreak time <arg7> number of damaged house <arg8> number of evacuated people <arg9> outbreak date <arg10> people/organization who predicted the disaster <arg11> refuge <arg12> the state of the volcano (dormant or active) <arg13> volcano name",
            "roles": ['Affected Areas', 'Casualties and Losses', 'Cause', 'Economic Loss', 'Fire Warning Level', 'Last Outbreak Time', 'Number of Damaged House', 'Number of Evacuated People', 'Outbreak Date', 'People/Organization who predicted the disaster', 'Refuge', 'The State of the Volcano (Dormant or Active)', 'Volcano Name'],
        },
        "Withdraw from an Orgnization": {
            "template": "<arg1> annoncement date <arg2> countries withdrawing from the organization <arg3> declarer <arg4> effective date <arg5> exit conditions <arg6> organization leader <arg7> organization members <arg8> organization name <arg9> withdraw reason",
            "roles": ['Annoncement Date', 'Countries Withdrawing from the Organization', 'Declarer', 'Effective Date', 'Exit Conditions', 'Organization Leader', 'Organization Members', 'Organization Name', 'Withdraw Reason'],
        },
    },
    "casie": {
        "Attack:Databreach": {
            "template": "<arg1> attack pattern <arg2> attacker <arg3> compromised data <arg4> damage amount <arg5> number of data <arg6> number of victim <arg7> place <arg8> purpose <arg9> time <arg10> tool <arg11> victim",
            "roles": ['Attack-Pattern', 'Attacker', 'Compromised-Data', 'Damage-Amount', 'Number-of-Data', 'Number-of-Victim', 'Place', 'Purpose', 'Time', 'Tool', 'Victim'],
        },
        "Attack:Phishing": {
            "template": "<arg1> attack pattern <arg2> attacker <arg3> damage amount <arg4> place <arg5> purpose <arg6> time <arg7> tool <arg8> trusted entity <arg9> victim",
            "roles": ['Attack-Pattern', 'Attacker', 'Damage-Amount', 'Place', 'Purpose', 'Time', 'Tool', 'Trusted-Entity', 'Victim'],
        },
        "Attack:Ransom": {
            "template": "<arg1> attack pattern <arg2> attacker <arg3> damage amount <arg4> payment method <arg5> place <arg6> price <arg7> time <arg8> tool <arg9> victim",
            "roles": ['Attack-Pattern', 'Attacker', 'Damage-Amount', 'Payment-Method', 'Place', 'Price', 'Time', 'Tool', 'Victim'],
        },
        "Vulnerability-related:DiscoverVulnerability": {
            "template": "<arg1> cve <arg2> capabilities <arg3> discoverer <arg4> supported platform <arg5> time <arg6> vulnerability <arg7> vulnerable system <arg8> vulnerable system owner <arg9> vulnerable system version",
            "roles": ['CVE', 'Capabilities', 'Discoverer', 'Supported_Platform', 'Time', 'Vulnerability', 'Vulnerable_System', 'Vulnerable_System_Owner', 'Vulnerable_System_Version'],
        },
        "Vulnerability-related:PatchVulnerability": {
            "template": "<arg1> cve <arg2> issues addressed <arg3> patch <arg4> patch number <arg5> releaser <arg6> supported platform <arg7> time <arg8> vulnerability <arg9> vulnerable system <arg10> vulnerable system version",
            "roles": ['CVE', 'Issues-Addressed', 'Patch', 'Patch-Number', 'Releaser', 'Supported_Platform', 'Time', 'Vulnerability', 'Vulnerable_System', 'Vulnerable_System_Version'],
        },
    },
    "mlee": {
        "Acetylation": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Binding": {
            "template": "<arg1> site <arg2> theme <arg3> theme 2",
            "roles": ['Site', 'Theme', 'Theme2'],
        },
        "Blood_vessel_development": {
            "template": "<arg1> at loc <arg2> from loc <arg3> theme",
            "roles": ['AtLoc', 'FromLoc', 'Theme'],
        },
        "Breakdown": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Catabolism": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Cell_division": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Cell_proliferation": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "DNA_methylation": {
            "template": "<arg1> site <arg2> theme",
            "roles": ['Site', 'Theme'],
        },
        "Death": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Dephosphorylation": {
            "template": "<arg1> site <arg2> theme",
            "roles": ['Site', 'Theme'],
        },
        "Development": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Dissociation": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Gene_expression": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Growth": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Localization": {
            "template": "<arg1> at loc <arg2> from loc <arg3> theme <arg4> to loc",
            "roles": ['AtLoc', 'FromLoc', 'Theme', 'ToLoc'],
        },
        "Metabolism": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Negative_regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme",
            "roles": ['CSite', 'Cause', 'Site', 'Theme'],
        },
        "Pathway": {
            "template": "<arg1> participant <arg2> participant 2 <arg3> participant 3 <arg4> participant 4",
            "roles": ['Participant', 'Participant2', 'Participant3', 'Participant4'],
        },
        "Phosphorylation": {
            "template": "<arg1> site <arg2> theme",
            "roles": ['Site', 'Theme'],
        },
        "Planned_process": {
            "template": "<arg1> instrument <arg2> instrument 2 <arg3> theme <arg4> theme 2",
            "roles": ['Instrument', 'Instrument2', 'Theme', 'Theme2'],
        },
        "Positive_regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme <arg5> theme 2",
            "roles": ['CSite', 'Cause', 'Site', 'Theme', 'Theme2'],
        },
        "Protein_processing": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme",
            "roles": ['CSite', 'Cause', 'Site', 'Theme'],
        },
        "Remodeling": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Reproduction": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Synthesis": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Transcription": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Translation": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Ubiquitination": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
    },
    "genia2011": {
        "Binding": {
            "template": "<arg1> site <arg2> site 2 <arg3> theme <arg4> theme 2 <arg5> theme 3 <arg6> theme 4",
            "roles": ['Site', 'Site2', 'Theme', 'Theme2', 'Theme3', 'Theme4'],
        },
        "Gene_expression": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Localization": {
            "template": "<arg1> at loc <arg2> theme <arg3> to loc",
            "roles": ['AtLoc', 'Theme', 'ToLoc'],
        },
        "Negative_regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme",
            "roles": ['CSite', 'Cause', 'Site', 'Theme'],
        },
        "Phosphorylation": {
            "template": "<arg1> site <arg2> theme",
            "roles": ['Site', 'Theme'],
        },
        "Positive_regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme",
            "roles": ['CSite', 'Cause', 'Site', 'Theme'],
        },
        "Protein_catabolism": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme",
            "roles": ['CSite', 'Cause', 'Site', 'Theme'],
        },
        "Transcription": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
    },
    "genia2013": {
        "Acetylation": {
            "template": "<arg1> site <arg2> theme",
            "roles": ['Site', 'Theme'],
        },
        "Binding": {
            "template": "<arg1> site <arg2> site 2 <arg3> theme <arg4> theme 2",
            "roles": ['Site', 'Site2', 'Theme', 'Theme2'],
        },
        "Deacetylation": {
            "template": "<arg1> cause <arg2> site <arg3> theme",
            "roles": ['Cause', 'Site', 'Theme'],
        },
        "Gene_expression": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Localization": {
            "template": "<arg1> theme <arg2> to loc",
            "roles": ['Theme', 'ToLoc'],
        },
        "Negative_regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme",
            "roles": ['CSite', 'Cause', 'Site', 'Theme'],
        },
        "Phosphorylation": {
            "template": "<arg1> cause <arg2> site <arg3> theme",
            "roles": ['Cause', 'Site', 'Theme'],
        },
        "Positive_regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme",
            "roles": ['CSite', 'Cause', 'Site', 'Theme'],
        },
        "Protein_catabolism": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Protein_modification": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Regulation": {
            "template": "<arg1> csite <arg2> cause <arg3> site <arg4> theme",
            "roles": ['CSite', 'Cause', 'Site', 'Theme'],
        },
        "Transcription": {
            "template": "<arg1> theme",
            "roles": ['Theme'],
        },
        "Ubiquitination": {
            "template": "<arg1> cause <arg2> site <arg3> theme",
            "roles": ['Cause', 'Site', 'Theme'],
        },
    },
    "muc4": {
        "Dummy": {
            "template": "<arg1> person <arg2> organization <arg3> target <arg4> victim <arg5> weapon",
            "roles": ['PerpInd', 'PerpOrg', 'Target', 'Victim', 'Weapon'],
        },
    },
}