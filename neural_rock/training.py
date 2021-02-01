import torch
from sklearn.metrics import f1_score


def train(sgd_steps, model, criterion, optimizer, loader, writer, scheduler=None, device="cuda"):
    model.train()

    running_loss = 0.0
    running_corrects = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        f1 = f1_score(labels.data.cpu(), preds.cpu(), average='macro')
        running_corrects += f1 * inputs.size(0)

        writer.add_scalar("f1_score", global_step=sgd_steps, scalar_value=f1)

        writer.add_scalar("loss", global_step=sgd_steps, scalar_value=loss.item())
        sgd_steps += 1

        if scheduler is not None:
            scheduler.step()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = running_corrects / len(loader.dataset)

    return sgd_steps, epoch_loss, epoch_f1


def validate(model, criterion, loader, device="cuda", return_predictions=False):
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    predictions = {"labels": [], "predictions": []}
    # Iterate over data.
    for inputs, labels in loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        labels = labels.cpu()
        preds = preds.cpu()

        if return_predictions:
            predictions["labels"] += list(labels.flatten().numpy())
            predictions["predictions"] += list(preds.flatten().numpy())

        # statistics
        running_loss += loss.item() * inputs.size(0)
        f1 = f1_score(labels.data, preds, average='macro')
        running_corrects += f1 * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = running_corrects / len(loader.dataset)

    return epoch_loss, epoch_f1, predictions