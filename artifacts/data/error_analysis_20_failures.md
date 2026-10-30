# Qualitative Error Analysis: 20 Failure Cases (Rank > 10)
This document analyzes 20 representative queries where the ground-truth passage was not retrieved within Top-10 by the final Hybrid + Cross-Encoder pipeline.
## Failure Taxonomy
1. **Type A: Candidate Generation Miss (Retrieval Bottleneck)** — True passage ranked > 30 in both BM25 and Dense, leaving Cross-Encoder unable to examine it.
2. **Type B: Vocabulary Mismatch / Jargon Polysemy** — Dense model matched general programming concepts rather than framework-specific subtleties.
3. **Type C: Non-Canonical Valid Answers** — The retrieved #1 passage provides an alternative working solution to the query, but carries a different StackOverflow `answer_id`.
4. **Type D: Code-Heavy Zero-Prose Passage** — Answer contains pure code snippets without explanatory narrative, hindering cross-attention text alignment.

---
## Detailed Inspection of 20 Failures

### Case #1: Query ID 80 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#>100` | Hybrid: `#>100` | **Final CE: `#>100`**
- **Query:**
> I've written a database generation script in SQL and want to execute it in my Adobe AIR application: Create Table tRole ( roleID integer Primary Key,roleName varchar(40) ); Create Table tFile ( fileID integer Primary Key,fileName varchar(50),fileDescription varchar(500),thumbnailID integer,fileForma...

- **Ground-Truth Passage (ID 124):**
> I wound up using this. It is a kind of a hack, but it actually works pretty well. The only thing is you have to be very careful with your semicolons.: D var strSql:String = stream.readUTFBytes(stream.bytesAvailable); var i:Number = 0; var strSqlSplit:Array = strSql.split(";"); for (i = 0; i < strSql...

- **Predicted Rank #1 Passage:**
> The problem is the scope of your temp table only exists within the exec() statement. You can transform your temp table into a "global" temp table by using 2 hash signs -> ##tempTable. However, I wonder why you are using a variable @dynamicallyBuiltWhereClause to generate your SQL statement. I have d...

---

### Case #2: Query ID 308 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#31` | Hybrid: `#31` | **Final CE: `#31`**
- **Query:**
> I often run into the following problem. I work on some changes to a project that require new tables or columns in the database. I make the database modifications and continue my work. Usually, I remember to write down the changes so that they can be replicated on the live system. However, I don't al...

- **Ground-Truth Passage (ID 360):**
> In Ruby on Rails, there's a concept of a migration -- a quick script to change the database. You generate a migration file, which has rules to increase the db version (such as adding a column) and rules to downgrade the version (such as removing a column). Each migration is numbered, and a table kee...

- **Predicted Rank #1 Passage:**
> See Is there a version control system for database structure changes? How do I version my MS SQL database in SVN? and Jeff's article Get Your Database Under Version Control I feel your pain, and I wish there were a better answer. This might be closer to what you were looking for. Mechanisms for trac...

---

### Case #3: Query ID 2123 (Category: Type C: Non-Canonical Valid Answers / Ranking Tie)
- **Rank Progression:** BM25: `#42` | Dense: `#20` | Hybrid: `#16` | **Final CE: `#17`**
- **Query:**
> Checkboxes in HTML forms don't have implicit labels with them. Adding an explicit label (some text) next to it doesn't toggle the checkbox. How do I make a checkbox toggle from clicking on the text label as well?...

- **Ground-Truth Passage (ID 2223):**
> Set the CSS display property for the label to be a block element and use that instead of your div - it keeps the semantic meaning of a label while allowing whatever styling you like. For example: label { width: 100px; height: 100px; display: block; background-color: #e0e0ff; } A ticky box!...

- **Predicted Rank #1 Passage:**
> Salut SnippetSpace! I prepared here a JSFiddle D E M O Hope this will help ciao et bonne chance!:) BTW, I used: $.fn.toggleChkBx = function() { if ($('.switch').attr('state') == 'on') { $(this).attr('checked',!$(this).find(':input').attr('checked')); } } And call it for Ex.: $('.some_Element').click...

---

### Case #4: Query ID 2311 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#>100` | Hybrid: `#>100` | **Final CE: `#>100`**
- **Query:**
> I have created a PHP-script to update a web server that is live inside a local directory. I'm migrating the script into Python. It works fine for the most part, but after a PUT command, the size of the file appears to change. Thus, the size of the file is different from that of the file on the serve...

- **Ground-Truth Passage (ID 2316):**
> Do you need to open the locfile in binary using rb? f = open (locfile, "rb")...

- **Predicted Rank #1 Passage:**
> Since PHP is a server-side language, the files you get using FTP_get() will end up on the server that is running the PHP script. You need to make sure that the directory on your server that is receiving the files has proper write permissions. I suggest CHMOD the dir to 777 to make sure that it works...

---

### Case #5: Query ID 4046 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#88` | Dense: `#52` | Hybrid: `#42` | **Final CE: `#42`**
- **Query:**
> I'm trying to convince my providers to use ANT instead of Rational Application Development so anyone can recompile, recheck, redeploy the solution anyplace, anytime, anyhow.:P I started a build.xml for a project that generates a JAR file but stopped there and I need real examples to compare notes. M...

- **Ground-Truth Passage (ID 4902):**
> My Environment: Fedora 8; WAS 6.1 (as installed with Rational Application Developer 7) The documentation is very poor in this area and there is a dearth of practical examples. Using the WebSphere Application Server (WAS) Ant tasks To run as described here, you need to run them from your server profi...

- **Predicted Rank #1 Passage:**
> You can use something like: so the ant task is propertyfile....

---

### Case #6: Query ID 4287 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#65` | Hybrid: `#68` | **Final CE: `#68`**
- **Query:**
> I have a process in erlang that is supposed to do something immediately after spawn, then send the result back to the parent when it is finished. How do I figure out the PID of the process that spawned it?...

- **Ground-Truth Passage (ID 4304):**
> You should pass self() to the child as one of the arguments to the entry function. spawn_link(?MODULE, child, [self()])....

- **Predicted Rank #1 Passage:**
> As far as I know a process doesn't have an explicit list of its children's PIDs, but it can easily be built, since a process should know which child processes it spawns. For example the UNIX fork() call returns the child PID in the parent process and 0 in the child process, CreateProcess() on Window...

---

### Case #7: Query ID 4738 (Category: Type C: Non-Canonical Valid Answers / Ranking Tie)
- **Rank Progression:** BM25: `#>100` | Dense: `#27` | Hybrid: `#27` | **Final CE: `#18`**
- **Query:**
> I'm developing a data access component that will be used in a website that contains a mix of classic ASP and ASP.NET pages, and need a good way to manage its configuration settings. I'd like to use a custom ConfigurationSection, and for the ASP.NET pages this works great. But when the component is c...

- **Ground-Truth Passage (ID 4746):**
> Try this: System.Configuration.ConfigurationFileMap fileMap = new ConfigurationFileMap(strConfigPath); //Path to your config file System.Configuration.Configuration configuration = System.Configuration.ConfigurationManager.OpenMappedMachineConfiguration(fileMap);...

- **Predicted Rank #1 Passage:**
> Well, somewhere in your code you must be creating the Configuration object and building the ISessionFactory. Get it from there......

---

### Case #8: Query ID 5222 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#>100` | Hybrid: `#>100` | **Final CE: `#>100`**
- **Query:**
> What is the Java equivalent of PHP's $_POST? After searching the web for an hour, I'm still nowhere closer....

- **Ground-Truth Passage (ID 5224):**
> Your HttpServletRequest object has a getParameter(String paramName) method that can be used to get parameter values. http://java.sun.com/javaee/5/docs/api/javax/servlet/ServletRequest.html#getParameter(java.lang.String)...

- **Predicted Rank #1 Passage:**
> Am I wrong or most of the answers are beside the point? To answer precisely your question, yes, you can submit a comma separated list to a POST parameter. To be honest, I just did a quick try with a PHP script, but I don't see why Java would behave differently. One point with POST requests is precis...

---

### Case #9: Query ID 5328 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#55` | Hybrid: `#55` | **Final CE: `#55`**
- **Query:**
> So, in Java, the first line of your constructor HAS to be a call to super... be it implicitly calling super(), or explicitly calling another constructor. What I want to know is, why can't I put a try block around that? My specific case is that I have a mock class for a test. There is no default cons...

- **Ground-Truth Passage (ID 5351):**
> Unfortunately, compilers can't work on theoretical principles, and even though you may know that it is safe in your case, if they allowed it, it would have to be safe for all cases. In other words, the compiler isn't stopping just you, it's stopping everyone, including all those that don't know that...

- **Predicted Rank #1 Passage:**
> If no explicit superclass constructor call is provided, and no call to a constructor in the same class is provided either, the no-args superclass constructor is always called. That's how Java is designed, and it would be too complicated and inefficient for the JVM to record which was the first const...

---

### Case #10: Query ID 5329 (Category: Type C: Non-Canonical Valid Answers / Ranking Tie)
- **Rank Progression:** BM25: `#1` | Dense: `#16` | Hybrid: `#6` | **Final CE: `#16`**
- **Query:**
> I'm currently evaluating the MSF for CMMI process template under TFS for use on my development team, and I'm having trouble understanding the need for separate bug and change request work item types. I understand that it is beneficial to be able to differentiate between bugs (errors) and change requ...

- **Ground-Truth Passage (ID 5403):**
> @Luke I don't disagree with you, but this difference is typically the explanation given for why there is two different processes available for handling the two types of issues. I'd say that if the color of the home page was originally designed to be red, and for some reason it is blue, that's easily...

- **Predicted Rank #1 Passage:**
> I think the principle of least surprise applies here: you need to be consistent with whatever you do. If you are unable to modify the URL scheme then that just leaves non-sequential ID's as the only viable solution. I don't personally see an issue with this, most bug trackers will be able to generat...

---

### Case #11: Query ID 5415 (Category: Type C: Non-Canonical Valid Answers / Ranking Tie)
- **Rank Progression:** BM25: `#>100` | Dense: `#11` | Hybrid: `#11` | **Final CE: `#12`**
- **Query:**
> I have a binary file that I have to parse and I'm using Python. Is there a way to take 4 bytes and convert it to a single precision floating point number?...

- **Ground-Truth Passage (ID 73281):**
> >>> import struct >>> struct.pack('f', 3.141592654) b'\xdb\x0fI@' >>> struct.unpack('f', b'\xdb\x0fI@') (3.1415927410125732,) >>> struct.pack('4f', 1.0, 2.0, 3.0, 4.0) '\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@'...

- **Predicted Rank #1 Passage:**
> For reference— future Python possibilities: Starting with Python 2.6 you can express binary literals using the prefix 0b or 0B: >>> 0b101111 47 You can also use the new bin function to get the binary representation of a number: >>> bin(173) '0b10101101' Development version of the documentation: What...

---

### Case #12: Query ID 5831 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#45` | Hybrid: `#46` | **Final CE: `#46`**
- **Query:**
> Does anyone have, or know of, a binary patch generation algorithm implementation in C#? Basically, compare two files (designated old and new ), and produce a patch file that can be used to upgrade the old file to have the same contents as the new file. The implementation would have to be relatively ...

- **Ground-Truth Passage (ID 5878):**
> Sorry I couldn't be more help. I would definately keep looking at xdelta because I have used it a number of times to produce quality diffs on 600MB+ ISO files we have generated for distributing our products and it performs very well....

- **Predicted Rank #1 Passage:**
> Let me start off with tried-and-true approaches, then point out some more recent approaches. approaches that I have seen work with binary files A long time ago, people expanded the old and the new versions of a binary file into temporary "text" files (every byte expanded to 3 bytes: 2 hex digits and...

---

### Case #13: Query ID 6957 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#43` | Hybrid: `#44` | **Final CE: `#44`**
- **Query:**
> Looking for feedback on: http://code.google.com/p/google-perftools/wiki/GooglePerformanceTools...

- **Ground-Truth Passage (ID 7458):**
> There's a pretty good post that I read a while back that outlines some testing and analysis of GPT in a variety of scenarios...

- **Predicted Rank #1 Passage:**
> You might find luck here http://code.google.com/p/gdata-objectivec-client/...

---

### Case #14: Query ID 7174 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#>100` | Hybrid: `#>100` | **Final CE: `#>100`**
- **Query:**
> Recently I had to develop a SharePoint workflow, and I found the experience quite honestly the most painful programming task I've ever had to tackle. One big problem I had was the problems I encountered when I had to step through it in the debugger. There's an article on how to debug a SharePoint wo...

- **Ground-Truth Passage (ID 7303):**
> The SharePoint team is currently working on MOSS extensions for VS 2008 which will allow this type of functionality. This was available in VS 2005 with MOSS extensions, but has to be run off Windows Server with a full MOSS installation and the correct permissions set....

- **Predicted Rank #1 Passage:**
> From Andrew Connell's blog post on the subject: Attaching the debugger to GAC'd assemblies: "Why aren't my breakpoints being hit?!?!" Ever been there? Me too... what a PITA that is! What's going on? Well, the assemblies are in the GAC and the Visual Studio debugger can't see the debugging symbols (a...

---

### Case #15: Query ID 7440 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#>100` | Hybrid: `#>100` | **Final CE: `#>100`**
- **Query:**
> The company I'm currently working for is using Selenium for Uniting-Testing our User Interface. What do you use to Unit-Test your Web UI and how effective do you find it?...

- **Ground-Truth Passage (ID 20754):**
> We use Watin at my place of employment, we are a.net shop so this solution made a lot of sense. We actually started with Watir (the original ruby implementation) and switched after. It's been a pretty good solution for us so far...

- **Predicted Rank #1 Passage:**
> UPDATE: Given WatiN has been stagnant for over a year now, I would direct anyone that needs web ui tests towards selenium, it is in continuous use & development by many contributors, and is actively used by Google. WatiN is the best that I've found. It integrates into Visual Studio unit testing or n...

---

### Case #16: Query ID 7885 (Category: Type B: Vocabulary Mismatch / Semantic Distraction)
- **Rank Progression:** BM25: `#>100` | Dense: `#22` | Hybrid: `#24` | **Final CE: `#24`**
- **Query:**
> Perl has OOP features, but they are somewhat rarely used. How do you create and use Perl objects with methods and properties?...

- **Ground-Truth Passage (ID 8668):**
> You should definitely take a look at Moose. package Point; use Moose; # automatically turns on strict and warnings

has 'x' => (is => 'rw', isa => 'Int'); has 'y' => (is => 'rw', isa => 'Int');

sub clear { my $self = shift; $self->x(0); $self->y(0); } Moose gives you (among other things) a construc...

- **Predicted Rank #1 Passage:**
> There are collection modules for more complex structures, but it is common style in Perl to use Arrays for arrays, stacks and lists. Perl has built in functions for using the array as a stack or list: push/pop, shift/unshift, splice (inserting or removing in the middle) and the foreach form for iter...

---

### Case #17: Query ID 8728 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#>100` | Hybrid: `#>100` | **Final CE: `#>100`**
- **Query:**
> I've got an MDI application written in Delphi 2006 which runs XP with the default theme. Is there a way of controlling the appearance of the MDI Children to avoid the large XP-style title bar on each window? I've tried setting the BorderStyle of the MDIChildren to bsSizeToolWin but they are still re...

- **Ground-Truth Passage (ID 161214):**
> All your need - overload procedure CreateWindowHandle, like this: unit CHILDWIN; interface uses Windows, Classes, Graphics, Forms, Controls, StdCtrls;

type TMDIChild = class(TForm) private { Private declarations } public { Public declarations } procedure CreateWindowHandle(const Params: TCreatePara...

- **Predicted Rank #1 Passage:**
> Setting FormBorderStyle = None will remove the title bar (at both design and run time) - and also remove your ability to resize the form. If you need a border you can set: ControlBox = false Text = ""...

---

### Case #18: Query ID 9002 (Category: Type A: Candidate Generation Miss (RRF Rank > 30))
- **Rank Progression:** BM25: `#>100` | Dense: `#42` | Hybrid: `#45` | **Final CE: `#45`**
- **Query:**
> When I use the sp_send_dbmail stored procedure, I get a message saying that my mail was queued. However, it never seems to get delivered. I can see them in the queue if I run this SQL: SELECT * FROM msdb..sysmail_allitems WHERE sent_status = 'unsent' This SQL returns a 1: SELECT is_broker_enabled FR...

- **Ground-Truth Passage (ID 9875):**
> Could be oodles of things. For example, I've seen (yes, actually seen) this happen after: Domain controller reboot Exchange server reboot Router outage Service account changes SQL Server running out of disk space So until it happens again, I wouldn't freak out over it....

- **Predicted Rank #1 Passage:**
> I found the problem... My MessageQueue was created with transaction true......

---

### Case #19: Query ID 10098 (Category: Type C: Non-Canonical Valid Answers / Ranking Tie)
- **Rank Progression:** BM25: `#>100` | Dense: `#12` | Hybrid: `#14` | **Final CE: `#12`**
- **Query:**
> I'm getting into more of a TDD workflow, and have a mix of MVC and asp.net Web Forms apps. MOQ is recommended for MVC. I've used Rhino for Web Forms. Does anyone have a best practice for having 1 framework mock for both?...

- **Ground-Truth Passage (ID 10192):**
> Rhino's latest release includes much of the sweet sweet 3.5 love that MoQ has. I'm a fan of MoQ, so that's what I'm using. But I also have Rhino, in case it does something that MoQ doesn't do. TL;DR: MoQ it baby....

- **Predicted Rank #1 Passage:**
> Using MoQ it looks something like this: var request = new Mock (); request.Expect(r => r.HttpMethod).Returns("GET"); var mockHttpContext = new Mock (); mockHttpContext.Expect(c => c.Request).Returns(request.Object); var controllerContext = new ControllerContext(mockHttpContext.Object, new RouteData(...

---

### Case #20: Query ID 10229 (Category: Type C: Non-Canonical Valid Answers / Ranking Tie)
- **Rank Progression:** BM25: `#>100` | Dense: `#15` | Hybrid: `#16` | **Final CE: `#17`**
- **Query:**
> Consider this problem: I have a program which should fetch (let's say) 100 records from a database, and then for each one it should get updated information from a web service. There are two ways to introduce parallelism in this scenario: I start each request to the web service on a new Thread. The n...

- **Ground-Truth Passage (ID 10484):**
> Option 3 is the best: Use Async IO. Unless your request processing is complex and heavy, your program is going to spend 99% of it's time waiting for the HTTP requests. This is exactly what Async IO is designed for - Let the windows networking stack (or.net framework or whatever) worry about all the ...

- **Predicted Rank #1 Passage:**
> Bad idea for lots of reasons. Others have mentioned some of them, but I'll give you one specific to your implementation: using ThreadStatic in ASP.NET will bite you eventually (see here ). You don't control the pipeline, so it's possible for multiple threads to end up servicing one request (think be...

---

